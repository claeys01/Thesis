# using Thesis

function train(; kws...)

    # load hyperparamters
    args = LuxArgs(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # load data and normalizer
    data, loaders, normalizer = @timeit to "get_data" get_data(
            args.batch_size,
            args.full_data_path;
            n_training = args.train_downsample,
            n_test = args.test_downsample,
            split = args.split,
            t_training = args.t_training,
        )
    TrainData, ValData, TestData = data
    train_loader, validation_loader, test_loader = loaders

    # if args.use_gpu
    #     device = gpu_device()
    # else
    #     device = cpu_device()
    # end

    device = get_device()

    normalizer = Normalizer(device(Float32.(normalizer.μ)), device(Float32.(normalizer.σ)), normalizer.method)

    @info "Training on $device"

    # initialize encoder and decoder
    if args.retrain 
        (enc, dec, ae, ps, st) = load_trained_AE(args.checkpoint_path; return_params=true, testmode=false)
        @info "Retraining model saved at $(args.checkpoint_path)"
    else
        enc = Encoder(args, verbose=true)
        dec = Decoder(args, verbose=true)
        ae = AE(enc, dec)
        rng = Xoshiro(args.seed)
        ps, st = Lux.setup(rng, ae) .|> device
        @info "AE initiated"
    end

    # define optimizer
    opt = AdamW(; eta=args.η, lambda=args.λ)

    train_state = Training.TrainState(ae, ps, st, opt)

    !ispath(args.save_path) && mkpath(args.save_path)

    loss_func = make_loss_function(args, device, normalizer)

    # Pre-allocate loss arrays
    max_iters = args.epochs * length(train_loader)
    train_losses = Vector{Float32}(undef, max_iters)
    rec_losses = Vector{Float32}(undef, max_iters)
    div_losses = Vector{Float32}(undef, max_iters)
    inside_losses = Vector{Float32}(undef, max_iters)
    iters = Vector{Int}(undef, max_iters)
    train_corrs = Vector{Vector{Float32}}(undef, max_iters)
    
    val_losses = Vector{Float32}(undef, args.epochs)
    val_iters = Vector{Int}(undef, args.epochs)
    val_corrs = Vector{Vector{Float32}}(undef, args.epochs)
    
    test_losses = Float32[]
    test_corrs = Vector{Float32}[]

    iter = 0

    # quick checks
    args.use_gpu && @info "CUDA.functional()" CUDA.functional()
    args.use_gpu && @info "CUDA device" CUDA.device()

    # training
    @info "Start Training, total $(args.epochs) epochs"
    val_mean = 1
    
    for epoch = 1:args.epochs
        epoch_start = time()
        @timeit to "epoch" begin
            @info "Epoch $(epoch)/$(args.epochs)"
            
            # train_progress = Progress(length(train_loader); desc="Training")
            # ---- TRAIN ----
            @timeit to "train" begin
                for train_idx in train_loader
                    batch = @timeit to "get training batch" build_batch(TrainData, train_idx) |> device
                    _, loss, stats, train_state = @timeit to "single_train_step!" Training.single_train_step!(
                        args.Autodiff, loss_func, batch, train_state; return_gradients=Val(false)
                    )
                    Lrec, Linside, L2div, corrs = stats
                    # record
                    iter += 1
                    iters[iter] = iter
                    train_losses[iter] = Float32(loss)
                    rec_losses[iter] = Float32(Lrec)
                    div_losses[iter] = Float32(L2div)
                    inside_losses[iter] = Float32(Linside)
                    train_corrs[iter] = corrs

                    # progress meter
                    # next!(train_progress; showvalues=[(:loss, loss)])
                end
            end
            # finish!(train_progress)

            # ---- VALIDATION (epoch-level) ----
            val_sum = 0.0
            n_val = 0
            val_corr_total = zeros(Float32, 2)
            @timeit to "validation" begin
                for val_idx in validation_loader
                    val_batch = @timeit to "get validation data" build_batch(ValData, val_idx) |> device
                    # Forward pass only (no gradients)
                    st_test = LuxCore.testmode(train_state.states)
                    val_loss, _, val_stats = @timeit to "validation loss" loss_func(ae, train_state.parameters, st_test, val_batch)
                    _, _, _, val_corr = val_stats
                    val_sum += val_loss
                    n_val += 1
                    val_corr_total .+= val_corr
                end
                val_mean = Float32(val_sum / max(n_val, 1))
                val_corr_mean = vec((val_corr_total / max(n_val, 1)))
                val_losses[epoch] = val_mean
                val_iters[epoch] = iter
                val_corrs[epoch] = val_corr_mean
            end

            # ----- TEST (On whole dataset)
            if args.test_loss
                @timeit to "test" begin
                    # test_progress = Progress(length(test_loader); desc="Test", color=:red)
                    test_sum = 0.0
                    n_test = 0
                    test_corr_mean = (0.0, 0.0)
                    test_corr_total = zeros(Float32, 2)
                    for test_idx in test_loader
                        test_batch = @timeit to "get test batch" build_batch(TestData, test_idx) |> device
                        # Forward pass only
                        st_test = LuxCore.testmode(train_state.states)
                        test_loss, _, test_stats = @timeit to "get test loss" loss_func(ae, train_state.parameters, st_test, test_batch)
                        _, _, _, test_corr = test_stats
                        test_sum += test_loss
                        n_test += 1
                        test_corr_total .+= test_corr
                        test_corr_mean = vec((test_corr_total / max(n_test, 1)))
                        # next!(test_progress; showvalues=[("Loss", test_loss), ("Corrs", test_corr_mean)])
                    end
                    test_mean = Float32(test_sum / max(n_test, 1))
                    push!(test_losses, test_mean)
                    push!(test_corrs, test_corr_mean)
                end
            end
        end
        # ---- epoch end: print concise summary ----
        epoch_time = time() - epoch_start
        test_corr_str = "($(round(test_corr_mean[1]; digits=3)), $(round(test_corr_mean[2]; digits=3)))"

        println(join([
            "Epoch $(epoch)/$(args.epochs)",
            "time=$(round(epoch_time; digits=3))s",
            "train_loss=$(round(train_losses[iter]; digits=4))",
            "train_corr=($(round(train_corrs[iter][1]; digits=3)), $(round(train_corrs[iter][2]; digits=3)))",
            "val_loss=$(round(val_mean; digits=4))",
            "val_corr=($(round(val_corr_mean[1]; digits=3)), $(round(val_corr_mean[2]; digits=3)))",
            "test_loss=$(round(test_mean; digits=4))",
            "test_corr=$(test_corr_str)"
        ], " | "))
    end

    # save model
    timestamp = Dates.format(now(), "udd-HHMM")
    tag = run_tag(args)
    save_folder = joinpath(args.save_path, "$(timestamp)__$(tag)")
    !ispath(save_folder) && mkpath(save_folder)
    filepath = joinpath(save_folder, "checkpoint.jld2")
    loss_trajectory_path = joinpath(save_folder, "loss_trajectory.jld2")


    let cpu = cpu_device()
        ps = cpu(train_state.parameters)
        st = cpu(train_state.states)
        args = struct2dict(args)
        JLD2.save(filepath,
            "ps", ps,
            "st", st,
            "normalizer", normalizer,
            "args", args,
        )

        JLD2.save(loss_trajectory_path,
            "train_losses", train_losses,
            "rec_losses", rec_losses,
            "div_losses", div_losses,
            "inside_losses", inside_losses,
            "iters", iters,
            "val_losses", val_losses,
            "val_iters", val_iters,
            "train_corrs", train_corrs,
            "val_corrs", val_corrs,
            "test_losses", test_losses,
            "test_corrs", test_corrs)
        @info "Model saved: $(filepath)"
    end

    #plot and save reconstruction of 2 random snapshots
    try
        reconstruction = visualize_reconstructions(filepath)
        reconstruct_path = joinpath(save_folder, "reconstruction.png")
        savefig(reconstruction, reconstruct_path)
        @info "Saved reconstruction plot to $reconstruct_path"
    catch e
        @warn "Failed to save reconstruction plot: $e"
    end

    try
        # plot and save loss evolution
        p = plot_losses(loss_trajectory_path, filepath)
        png_path = joinpath(save_folder, "loss_evolution.png")
        savefig(p, png_path)
        @info "Saved loss plot to $png_path"
    catch e
        @warn "Failed to save loss plot: $e"
    end
    show(to)
end


_fmt(x) = replace(string(round(Float64(x), sigdigits=3)), "." => "p")
_yn(b) = b ? "Y" : "N"

function run_tag(args)
    H, W, Cin = args.input_dim
    _, _, Cout = args.output_dim

    return join([
        "E$(args.epochs)",
        "HW$(H)x$(W)",
        "C$(Cin)to$(Cout)",
        "nc$(args.n_conv)",
        "nd$(args.n_dense)",
        "z$(args.latent_dim)",
        "C$(args.C_base)",
        "lr$(_fmt(args.η))",
        "wd$(_fmt(args.λ))",
        "bs$(args.batch_size)",
        "N$(_yn(args.normalize))",
        "L$(args.loss)",
    ], "_")
end

function make_loss_function(args, device, normalizer)
    function loss_function(m, ps, st, batch)
        x_in, x_target, μ₀ = batch

        # Move to GPU/CPU
        x_in_dev = device(x_in)
        x_target_dev = device(x_target)
        μ₀_dev = device(μ₀)

        # Optional normalization
        if args.normalize
            uvmask = x_in_dev
            uvc = uvmask[:, :, 1:2, :]
            uvc_norm, _ = normalize_batch(uvc; normalizer=normalizer)
            x_in_dev = cat(uvc_norm, uvmask[:, :, 3:4, :]; dims=3)
            x_target_dev, _ = normalize_batch(x_target_dev; normalizer=normalizer)
        end

        # Call your existing loss
        return total_loss(
            m, ps, st,
            x_in_dev, x_target_dev, μ₀_dev;
            loss=args.loss,
            λdiv=args.λdiv,
            λmask=args.λmask,
        )
    end
    return loss_function

end





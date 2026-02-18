# using Thesis

function train_AE(args::LuxArgs)
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

    max_iters = args.epochs * length(train_loader)
    
    train_losses  = Float32[]; sizehint!(train_losses, max_iters)
    rec_losses    = Float32[]; sizehint!(rec_losses, max_iters)
    div_losses    = Float32[]; sizehint!(div_losses, max_iters)
    curl_losses   = Float32[]; sizehint!(curl_losses, max_iters)
    strain_losses = Float32[]; sizehint!(strain_losses, max_iters)
    iters         = Int[];     sizehint!(iters, max_iters)
    train_corrs   = Vector{Float32}[]; sizehint!(train_corrs, max_iters)
    
    val_losses = Float32[]; sizehint!(val_losses, args.epochs)
    val_iters  = Int[];     sizehint!(val_iters, args.epochs)
    val_corrs  = Vector{Float32}[]; sizehint!(val_corrs, args.epochs)
    
    test_losses = Float32[]
    test_corrs  = Vector{Float32}[]
    test_Lrecs = Float32[]
    iter = 0

    # quick checks
    (USE_CUDA[] & args.use_gpu) && @info "CUDA.functional()" CUDA.functional()
    (USE_CUDA[] & args.use_gpu) && @info "CUDA device" CUDA.device()

    # training
    @info "Start Training, total $(args.epochs) epochs"
    val_mean = 1
    
    for epoch = 1:args.epochs
        epoch_start = time()
        @timeit to "epoch" begin
            @info "Epoch $(epoch)/$(args.epochs)"
            
            # ---- TRAIN ----
            @timeit to "train" begin
                for train_idx in train_loader
                    batch = @timeit to "get training batch" build_batch(TrainData, train_idx) |> device
                    _, loss, stats, train_state = @timeit to "single_train_step!" Training.single_train_step!(
                        args.Autodiff, loss_func, batch, train_state; return_gradients=Val(false)
                    )
                    Lrec, Ldiv, Lcurl, Lstrain, corrs = stats
                    # record
                    iter += 1
                    push!(iters, iter)
                    push!(train_losses, Float32(loss))
                    push!(rec_losses, Float32(Lrec))
                    push!(div_losses, Float32(Ldiv))
                    push!(curl_losses, Float32(Lcurl))
                    push!(strain_losses, Float32(Lstrain))
                    push!(train_corrs, corrs)
                end
            end

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
                    _, _, _, _, val_corr = val_stats
                    val_sum += val_loss
                    n_val += 1
                    val_corr_total .+= val_corr
                end
                val_mean = Float32(val_sum / max(n_val, 1))
                val_corr_mean = vec((val_corr_total / max(n_val, 1)))
                push!(val_losses, val_mean)
                push!(val_iters, iter)
                push!(val_corrs, val_corr_mean)
            end

            # ----- TEST (On whole dataset)
            if args.test_loss
                @timeit to "test" begin
                    test_sum = 0.0
                    test_Lrec_sum = 0.0
                    n_test = 0
                    test_corr_mean = (0.0, 0.0)
                    test_corr_total = zeros(Float32, 2)
                    for test_idx in test_loader
                        test_batch = @timeit to "get test batch" build_batch(TestData, test_idx) |> device
                        # Forward pass only
                        st_test = LuxCore.testmode(train_state.states)
                        test_loss, _, test_stats = @timeit to "get test loss" loss_func(ae, train_state.parameters, st_test, test_batch)
                        test_Lrec, _, _, _, test_corr = test_stats
                        
                        test_sum += test_loss
                        test_Lrec_sum += test_Lrec
                        n_test += 1
                        test_corr_total .+= test_corr
                        test_corr_mean = vec((test_corr_total / max(n_test, 1)))
                    end
                    test_mean = Float32(test_sum / max(n_test, 1))
                    test_Lrec_mean = Float32(test_Lrec_sum / max(n_test, 1))
                    push!(test_losses, test_mean)
                    push!(test_corrs, test_corr_mean)
                    push!(test_Lrecs, test_Lrec_mean)
                end
            end
        end
        # ---- epoch end: print concise summary ----
        epoch_time = time() - epoch_start
        test_corr_str = "($(round(test_corr_mean[1]; digits=3)), $(round(test_corr_mean[2]; digits=3)))"

        println(join([
            "Epoch $(epoch)/$(args.epochs)",
            "time=$(round(epoch_time; digits=3))s",
            "train_loss=$(round(train_losses[end]; digits=4))",
            "train_corr=($(round(train_corrs[end][1]; digits=3)), $(round(train_corrs[end][2]; digits=3)))",
            "val_loss=$(round(val_mean; digits=4))",
            "val_corr=($(round(val_corr_mean[1]; digits=3)), $(round(val_corr_mean[2]; digits=3)))",
            "test_loss=$(round(test_mean; digits=4))",
            "test_corr=$(test_corr_str)"
        ], " | "))
    end

    # save model
    timestamp = Dates.format(now(), "udd-HHMM")
    tag = run_tag(args; test_Lrec=test_Lrecs[end])
    save_folder = joinpath(args.save_path, "$(timestamp)__$(tag)")
    !ispath(save_folder) && mkpath(save_folder)
    filepath = joinpath(save_folder, "checkpoint.jld2")
    loss_trajectory_path = joinpath(save_folder, "loss_trajectory.jld2")


    let cpu = cpu_device()
        # Always save on CPU for portability
        ps_cpu = cpu(train_state.parameters)
        st_cpu = cpu(train_state.states)

        normalizer_cpu = Normalizer(
        Array(normalizer.μ),
        Array(normalizer.σ),
        normalizer.method
    )
        args = struct2dict(args)
        JLD2.save(filepath,
            "ps", ps_cpu,
            "st", st_cpu,
            "normalizer", normalizer_cpu,
            "args", args,
        )

        JLD2.save(loss_trajectory_path,
            "train_losses", train_losses,
            "rec_losses", rec_losses,
            "div_losses", div_losses,
            "curl_losses", curl_losses,
            "strain_losses", strain_losses,
            "iters", iters,
            "train_corrs", train_corrs,

            "val_losses", val_losses,
            "val_iters", val_iters,
            "val_corrs", val_corrs,

            "test_losses", test_losses,
            "test_Lrecs", test_Lrecs,
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

function run_tag(args; test_Lrec=0.0)
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
        "Tl$(_fmt(test_Lrec))"
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
            # x_target_dev, _ = normalize_batch(x_target_dev; normalizer=normalizer)
        end

        # Call your existing loss
        return total_loss(
            m, ps, st, normalizer, 
            x_in_dev, x_target_dev, μ₀_dev;
            loss=args.loss,
            λdiv=args.λdiv,
            λstrain=args.λstrain,
            λcurl=args.λcurl
        )
    end
    return loss_function

end





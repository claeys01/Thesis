using JLD2
using Lux
using Random
using ProgressMeter
using Plots
using Dates
using Optimisers
using Enzyme
using Zygote
using DrWatson: struct2dict

includet("../custom.jl")
includet("Lux_AE.jl")
includet("../utils/Lux_AE_reconstructer.jl")
includet("../utils/Lux_AE_loss_plot.jl")



function train(; kws...)

    # load hyperparamters
    args = LuxArgs(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # load RHS data and normalizer
    train_loader, validation_loader, normalizer = get_data(args.batch_size, args.data_path;
        n_samples=args.downsample, clip_bc=args.clip_bc, split=args.split)

    if args.use_gpu
        device = gpu_device()

    else
        device = cpu_device()
    end

    normalizer = Normalizer(device(Float32.(normalizer.μ)), device(Float32.(normalizer.σ)), normalizer.method)

    @info "Training on $device"

    # # initialize encoder and decoder
    encoder = Encoder(args.input_dim,  args.latent_dim; hidden_dim=args.hidden_dim, C_next=args.C_conv, padding=args.padding, stride=args.stride)
    decoder = Decoder(args.output_dim, args.latent_dim; hidden_dim=args.hidden_dim, C_next=args.C_conv) 

    ae = AE(encoder, decoder)

    rng = Xoshiro(args.seed)
    ps, st = Lux.setup(rng, ae) .|> device

    # define optimizer
    opt = AdamW(; eta=args.η, lambda=args.λ)

    train_state = Training.TrainState(ae, ps, st, opt)

    !ispath(args.save_path) && mkpath(args.save_path)

    loss_func = make_loss_function(args, device, normalizer)

    # record losses
    train_losses = Float32[]
    val_losses = Float32[]
    rec_losses = Float32[]
    div_losses = Float32[]
    inside_losses = Float32[]
    iters = Int[]
    val_iters = Int[]
    train_corrs = Vector{Float32}[]
    val_corrs = Vector{Float32}[]
    iter = 0

    # training
    @info "Start Training, total $(args.epochs) epochs"
    val_mean = 1
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)/$(args.epochs)"
        train_progress = Progress(length(train_loader); desc="Training")
        # ---- TRAIN ----
        for batch in train_loader
            x_in, x_target, μ₀ = batch
            _, loss, stats, train_state = Training.single_train_step!(
                args.Autodiff, loss_func, batch, train_state; return_gradients=Val(false)
            )
            Lrec, Linside, L2div, corrs = stats

            # x_lux, _ = ae(x_in_dev, train_state.parameters, train_state.states)                         # Lux version, same F32/device
            # @show mean(x_lux)
            # record
            iter += 1
            push!(iters, iter)
            push!(train_losses, Float32(loss))
            push!(rec_losses, Float32(Lrec))
            push!(div_losses, Float32(L2div))
            push!(inside_losses, Float32(Linside))
            push!(train_corrs, corrs)


            # progress meter
            next!(train_progress; showvalues=[(:loss, loss)])
        end

        finish!(train_progress)

        # ---- VALIDATION (epoch-level) ----
        val_sum = 0.0
        n_val = 0
        val_corr_total = zeros(Float32, 2)


        for val_batch in validation_loader
            _, val_loss, val_stats, train_state = Training.single_train_step!(
                args.Autodiff, loss_func, val_batch, train_state; return_gradients=Val(false)
            )

            _, _, _, val_corr = val_stats
            val_sum += val_loss
            n_val += 1
            val_corr_total .+= val_corr
        end
        val_mean = Float32(val_sum / max(n_val, 1))
        val_corr_mean = vec((val_corr_total / max(n_val, 1)))
        push!(val_losses, val_mean)
        push!(val_iters, iter)      # <— align the validation point to the last train iter of this epoch
        push!(val_corrs, val_corr_mean)
    end

    # save model
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    save_folder = joinpath(args.save_path, timestamp)
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
            "val_corrs", val_corrs)

        @info "Model saved: $(filepath)"
    end

    #plot and save reconstruction of 2 random snapshots
    try
        # reconstruction = visualize_reconstructions(;encoder=encoder, decoder=decoder, args=args)
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


if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    train()
    # run_dim_check()
end
# train()


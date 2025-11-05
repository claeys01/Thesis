using JLD2
using Lux
using Optimisers: AdamW
using WaterLily
using Random
using ProgressMeter
using Plots
using Dates
# using CUDA
using DrWatson: struct2dict

includet("../custom.jl")
includet("AE_core.jl")
includet("../utils/AE_reconstructer.jl")
includet("../utils/AE_loss_plot.jl")



function train(; kws...)

    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # load RHS data and normalizer
    train_loader, validation_loader, normalizer = get_data(args.batch_size, args.data_path;
        n_samples=args.downsample, clip_bc=args.clip_bc, split=args.split)

    if args.use_gpu
        device = cpu_device()

    else
        device = CUDADevice()
    end

    normalizer = Normalizer(device(Float32.(normalizer.μ)), device(Float32.(normalizer.σ)), normalizer.method)

    @info "Training on $device"


    # # initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride) |> device
    decoder = Decoder(args.output_dim, args.latent_dim; C_next=args.C_conv) |> device

    enc_

    # define optimizer
    opt_enc = Flux.setup(AdamW(eta=args.η, lambda=args.λ), encoder)
    opt_dec = Flux.setup(AdamW(eta=args.η, lambda=args.λ), decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

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
        @info "Epoch $(epoch)"
        train_progress = Progress(length(train_loader); desc="Training")
        # ---- TRAIN ----
        for (x_in, x_target, μ₀) in train_loader
            # println(size(x_in))
            loss_tuple, grad_enc, grad_dec = training_step(encoder, decoder, x_in, x_target, μ₀, device, args, normalizer)
            loss_total, (Lrec, Linside, L2div), corrs = loss_tuple

            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)

            # record
            iter += 1
            push!(iters, iter)
            push!(train_losses, Float32(loss_total))
            push!(rec_losses, Float32(Lrec))
            push!(div_losses, Float32(L2div))
            push!(inside_losses, Float32(Linside))
            push!(train_corrs, corrs)


            # progress meter
            next!(train_progress; showvalues=[(:loss, loss_total)])
        end

        finish!(train_progress)

        # ---- VALIDATION (epoch-level) ----
        val_sum = 0.0
        n_val = 0
        val_corr_total = zeros(Float32, 2)


        for (x_in_val, x_target_val, μ₀_val) in validation_loader
            val_loss_tuple, _, _ = training_step(encoder, decoder, x_in_val, x_target_val, μ₀_val, device, args, normalizer)
            val_loss_total, _, val_corr = val_loss_tuple
            val_sum += val_loss_total
            n_val += 1
            val_corr_total .+= val_corr
        end
        # println(val_)
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


    let encoder = cpu(encoder), decoder = cpu(decoder), args = struct2dict(args)
        JLD2.save(filepath, "encoder", Flux.state(encoder),
            "decoder", Flux.state(decoder),
            "normalizer", normalizer,
            "args", args)
        JLD2.save(loss_trajectory_path, "train_losses", train_losses,
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

function training_step(encoder, decoder, x_in, x_target, μ₀, device, args, normalizer)
    # # Move to GPU/CPU
    x_in_dev = device(x_in)
    x_target_dev = device(x_target)
    μ₀_dev = device(μ₀)

    # # Optional normalization
    if args.normalize
        # x_in_dev = (u,v,mask)
        # normalize u,v, leave mask alone:
        uvmask = x_in_dev
        uvc = uvmask[:, :, 1:2, :]                # u,v
        uvc_norm, _ = normalize_batch(uvc; normalizer=normalizer)

        x_in_dev = cat(uvc_norm, uvmask[:, :, 3:4, :]; dims=3)
        x_target_dev, _ = normalize_batch(x_target_dev; normalizer=normalizer)

    end
    # println(size(x_target_dev))
    loss_tuple, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
        total_loss(enc, dec,
            x_in_dev,
            x_target_dev,
            μ₀_dev;
            loss=args.loss,
            λdiv=args.λdiv,
            λmask=args.λmask)
    end


    return loss_tuple, grad_enc, grad_dec
end



if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    train()
    # run_dim_check()
end
# train()


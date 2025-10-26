using JLD2
using Flux
using Optimisers: AdamW
using WaterLily
using Random
using ProgressMeter: Progress, next!
using Plots
using Dates
using DrWatson: struct2dict

includet("../custom.jl")
includet("AE_core.jl")
includet("../utils/AE_reconstructer.jl")



function train(; kws...)

    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # load RHS data and normalizer
    loader, normalizer = get_data(args.batch_size, args.data_path; n_samples=args.downsample, normalize=args.normalize, clip_bc=args.clip_bc)

    if args.use_gpu
        device = Flux.get_device()
        normalizer = Normalizer(normalizer.μ |> device, normalizer.σ |> device, normalizer.method)

    else
        device = Flux.get_device("CPU")
    end

    @info "Training on $device"

 
    # initialize encoder and decoder
    encoder = Flux.f32(Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride)) |> device
    decoder = Flux.f32(Decoder(args.input_dim, args.latent_dim; C_next=args.C_conv)) |> device

    # define optimizer
    opt_enc = Flux.setup(AdamW(eta=args.η, lambda=args.λ), encoder)
    opt_dec = Flux.setup(AdamW(eta=args.η, lambda=args.λ), decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # record losses
    train_losses = Float32[]
    rec_losses = Float32[]
    div_losses = Float32[]
    div_diff_losses = Float32[]
    iters = Int[]
    iter = 0

    # training
    @info "Start Training, total $(args.epochs) epochs"

    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))
        for (i, x) in enumerate(loader)
            x_dev = x |> device
            if args.normalize
                x_dev, _ = normalize_batch(x_dev; normalizer=normalizer)    
            end

            loss_tuple, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                    total_loss(enc, dec, x; λdiv=args.λdiv, λdiff=args.λdiff)
            end
            loss_total, (Lrec, L2div, L2div_diff) = loss_tuple

            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)

            # record
            iter += 1
            push!(iters, iter)
            push!(train_losses, Float32(loss_total))
            push!(rec_losses, Float32(Lrec))
            push!(div_losses, Float32(L2div))
            push!(div_diff_losses, Float32(L2div_diff))

            # progress meter
            next!(progress; showvalues=[(:loss, loss_total, (Lrec, L2div, L2div_diff))]) 
        end
    end
    
    # save model
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    save_folder = joinpath(args.save_path, timestamp)
    !ispath(save_folder) && mkpath(save_folder)
    filepath = joinpath(save_folder, "checkpoint.jld2")
    loss_trajectory_path = joinpath(save_folder, "loss_trajectory.jld2")


    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        JLD2.save(filepath, "encoder", Flux.state(encoder),
                            "decoder", Flux.state(decoder),
                            "normalizer", normalizer,
                            "args", args)
        JLD2.save(loss_trajectory_path, "train_losses", train_losses,
                                        "rec_losses", rec_losses,
                                        "div_losses", div_losses,
                                        "div_diff_losses", div_diff_losses,
                                        "iters", iters)
                                                     
        @info "Model saved: $(filepath)"
    end

    #plot and save reconstruction of 2 random snapshots
    try 
        reconstruction = visualize_reconstructions(;encoder=encoder, decoder=decoder, args=args)
        reconstruct_path = joinpath(save_folder, "reconstruction.png")
        savefig(reconstruction, reconstruct_path)
        @info "Saved reconstruction plot to $reconstruct_path"
    catch e
        @warn "Failed to save reconstruction plot: $e"
    end

    # plot and save loss evolution
    try
        p = plot(iters, train_losses, label="total", xlabel="Iteration", ylabel="Loss", title="Training loss", lw=2)
        plot!(p, iters, rec_losses, label="reconstruction", lw=1, ls=:dash)
        if args.λdiv != 0
            plot!(p, iters, div_losses, label="divergence", lw=1, ls=:dot)
        end
        if args.λdiff != 0
            plot!(p, iters, div_diff_losses, label="divergence difference", lw=1, ls=:dashdot)
        end
        png_path = joinpath(save_folder, "loss_evolution.png")
        savefig(p, png_path)
        @info "Saved loss plot to $png_path"
    catch e
        @warn "Failed to save loss plot: $e"
    end
end


if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    train()
    # run_dim_check()
end
# train()


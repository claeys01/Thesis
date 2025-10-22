using JLD2
using CUDA
using Flux
using Flux: glorot_uniform, Conv, ConvTranspose, Dense, Chain, relu, MaxPool
using Optimisers: AdamW
using WaterLily
using Random
using Statistics
using ProgressMeter: Progress, next!
using MLUtils: DataLoader
using Zygote
using Plots
using Dates
using DrWatson: struct2dict
using ChainPlots

includet("../custom.jl")
includet("AE_core.jl")
includet("../utils/AE_reconstructer.jl")


function train(; kws...)

    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.use_gpu
        device = Flux.get_device()
    else
        device = Flux.get_device("CPU")
    end

    @info "Training on $device"

    # load RHS data
    loader = get_data(args.batch_size, args.data_path; n_samples=args.downsample)

    # initialize encoder and decoder
    encoder = Flux.f32(Encoder(args.input_dim, args.latent_dim)) |> device
    decoder = Flux.f32(Decoder(args.input_dim, args.latent_dim)) |> device

    # define optimizer
    opt_enc = Flux.setup(AdamW(eta=args.η, lambda=args.λ), encoder)
    opt_dec = Flux.setup(AdamW(eta=args.η, lambda=args.λ), decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # record losses
    train_losses = Float32[]
    rec_losses = Float32[]
    div_losses = Float32[]
    iters = Int[]
    iter = 0

    # training
    @info "Start Training, total $(args.epochs) epochs"

    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))
        for (i, x) in enumerate(loader)
            x_dev = x |> device            

            # capture both total loss and components
            loss_tuple, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                total_loss(enc, dec, x_dev)
            end
            loss_total, (Lrec, L2div) = loss_tuple

            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)

            # record
            iter += 1
            push!(iters, iter)
            push!(train_losses, Float32(loss_total))
            push!(rec_losses, Float32(Lrec))
            push!(div_losses, Float32(L2div))

            # progress meter
            next!(progress; showvalues=[(:loss, loss_total)]) 
        end
    end
    # save model

    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    save_folder = joinpath(args.save_path, timestamp)
    !ispath(save_folder) && mkpath(save_folder)
    filepath = joinpath(save_folder, "checkpoint.jld2")


    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        JLD2.save(filepath, "encoder", Flux.state(encoder),
                            "decoder", Flux.state(decoder),
                            "args", args)                            
        @info "Model saved: $(filepath)"
    end

    #plot and save reconstruction of 2 random snapshots
    try 
        reconstruction = visualize_reconstructions(filepath; n=args.n_reconstruct)
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
        if any(!iszero, div_losses)
            plot!(p, iters, div_losses, label="divergence", lw=1, ls=:dot)
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


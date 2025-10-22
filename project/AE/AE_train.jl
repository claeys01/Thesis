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

    # load RHS data (collect full dataset to compute normalizer)
    raw_loader = get_data(args.batch_size, args.data_path; n_samples=args.downsample)

    # collect all batches into one array (H,W,C,N_total)
    Xparts = Any[]
    for b in raw_loader
        push!(Xparts, b)
    end
    Xall = cat(Xparts...; dims=4)

    # compute normalizer from entire dataset (on CPU)
    _, normalizer = normalize_batch(Xall; normalizer=nothing)

    # move normalizer to device if using GPU
    if args.use_gpu
        device = Flux.get_device()
        normalizer = Normalizer(normalizer.μ |> device, normalizer.σ |> device, normalizer.method)
    else
        device = Flux.get_device("CPU")
    end

    # recreate loader from full dataset so we can iterate fresh during training
    loader = DataLoader(Xall, batchsize=args.batch_size, shuffle=true)

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

            # normalize batch using precomputed normalizer (x_norm keeps same device)
            x_norm, _ = normalize_batch(x_dev; normalizer=normalizer)

            # compute loss & gradients using normalized data; denormalize only for divergence terms
            loss_tuple, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                ŷ_norm = reconstruct(enc, dec, x_norm)
                Lrec = recon_loss(ŷ_norm, x_norm)

                L2div = zero(eltype(Lrec))
                L2div_diff = zero(eltype(Lrec))

                # denormalize predictions / targets only when computing divergence-based losses
                if args.λdiv != 0
                    try
                        ŷ_den = denormalize_batch(ŷ_norm, normalizer)
                        L2div = div_loss_L2(ŷ_den)
                    catch e
                        @warn "divergence loss (λdiv) failed; skipping divergence term: $e"
                        L2div = zero(eltype(Lrec))
                    end
                end

                if args.λdiff != 0
                    try
                        ŷ_den = denormalize_batch(ŷ_norm, normalizer)
                        x_den = denormalize_batch(x_norm, normalizer)
                        L2div_diff = div_diff_loss(ŷ_den, x_den)
                    catch e
                        @warn "divergence-diff loss (λdiff) failed; skipping divergence-diff term: $e"
                        L2div_diff = zero(eltype(Lrec))
                    end
                end

                return Lrec + args.λdiv * L2div + args.λdiff * L2div_diff, (Lrec, L2div, L2div_diff)
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


    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args), normalizer=Normalizer(cpu(normalizer.μ), cpu(normalizer.σ), normalizer.method)
        JLD2.save(filepath, "encoder", Flux.state(encoder),
                            "decoder", Flux.state(decoder),
                            "args", args,
                            "normalizer", normalizer)                            
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


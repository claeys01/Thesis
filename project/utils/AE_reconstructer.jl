using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/AE_core.jl")
includet("../custom.jl")
includet("../utils/AE_normalizer.jl")



function visualize_reconstructions(checkpoint_path::Union{String,Nothing}=nothing;
                                   device=Flux.get_device("CPU"))
    """
    Visualize reconstructions.

    Usage:
      - provide checkpoint_path (String) -> load encoder/decoder and args from checkpoint
    """
    # load from checkpoint

    checkpoint = JLD2.load(checkpoint_path)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    normalizer = checkpoint["normalizer"]
    args = Args(; checkpoint["args"]...)
    enc = Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride, verbose=false)
    dec = Decoder(args.output_dim, args.latent_dim; C_next=args.C_conv, verbose=false)
    Flux.loadmodel!(enc, encoder_state)
    Flux.loadmodel!(dec, decoder_state)

    # optional seeding
    args.seed > 0 && Random.seed!(args.seed)

    @load args.data_path data
    preprocess_data!(data; n_samples=args.downsample, clip_bc=args.clip_bc,verbose=false)


    span = length(data["time"])

    if span < args.n_reconstruct
        error("amount of data must be ≥ $(args.n_reconstruct)")
    end

    # sample without replacement from the actual span
    ids = randperm(span)[1:args.n_reconstruct]


    @info "Selected $(args.n_reconstruct) $(args.field) snapshots with indices $ids for reconstruction"
    # prepare plotting grid: each sample has C rows; three columns (input, recon, colorbar-only)
    plots = []
    dirs = ["x" , "y"]
    for (s, id) in enumerate(ids)
        μ₀ = data["μ₀"][:, :, :, id:id]
        x = data[args.field][:, :, :, id:id]
        if args.normalize
            x_target, _ = normalize_batch(x, normalizer=normalizer)
            x_in = cat(x_target, μ₀; dims=3)
            x̂_norm = reconstruct(enc, dec, x_in)
            x̂ = denormalize_batch(x̂_norm, normalizer)
            x_target = denormalize_batch(x_target, normalizer)
        else
            x_in = cat(x, μ₀; dims=3)
            x̂ = reconstruct(enc, dec, x_in, μ₀)
            # ensure x_target exists for plotting (use original input snapshot)
            x_target = x
        end


        for ch in 1:2
            mat_in = x_target[:, :, ch] 
            mat_out = x̂[:, :, ch] .* μ₀[:, :, ch]


            μ = mean(mat_in)
            σ = std(mat_in)
            clim = (μ - σ, μ + σ)

            # left plot (input): no colorbar
            img_in = flood(mat_in;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title="snapshot $(ids[s]): Input $(dirs[ch])",
                titlefontsize=8)

            # right plot (reconstructed): only here show colorbar
            img_out = flood(mat_out;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title="snapshot $(ids[s]): Reconstructed $(dirs[ch])",
                titlefontsize=8)

            # third column: colorbar-only plot
            # create a tiny image whose colorbar uses the same clims; hide axes so only the colorbar is visible
            cb_sample = fill(μ, 1, 1)
            push!(plots, img_in)
            push!(plots, img_out)
        end
    end
    
    p = plot(plots...;
             layout=(args.n_reconstruct*2, 2),
             link=:none, legend=false,
             size=(500, 750),
             dpi=200, grid=false)

    return p
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    checkpoint = "data/models/2025-11-11_16-20-27/checkpoint.jld2"
    # visualize_reconstructions(checkpoint)

end


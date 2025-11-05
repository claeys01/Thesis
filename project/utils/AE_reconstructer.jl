using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/AE_core.jl")
includet("../custom.jl")
includet("../utils/AE_normalizer.jl")



function visualize_reconstructions(checkpoint_path::Union{String,Nothing}=nothing;
                                   encoder=nothing, decoder=nothing,
                                   args::Union{Args,Nothing}=nothing, 
                                   device=Flux.get_device("CPU"))
    """
    Visualize reconstructions.

    Usage:
      - provide checkpoint_path (String) -> load encoder/decoder and args from checkpoint
      - OR provide encoder and decoder objects and args (Args) -> use those
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

    # @load args.data_path RHS_data
    snapshots, _, ids = get_random_snapshots(args.data_path; n=args.n_reconstruct, downsample=args.downsample)

    @load args.data_path RHS_data

    _, _, C, nn = size(snapshots)
    @info "Selected $nn snapshots indices for reconstruction: $ids"
    # prepare plotting grid: each sample has C rows; two columns (input, recon)
    plots = []
    dirs = ["x" , "y"]
    for s in 1:nn
        μ₀ = remove_ghosts(RHS_data["μ₀"][ids[s]])
        println(size(μ₀))
        if args.normalize
            x_target, _ = normalize_batch(snapshots[:, :, :, s:s], normalizer=normalizer)
            x_in = cat(x_target, μ₀; dims=3) 
            x̂_norm = reconstruct(enc, dec, x_in, μ₀)
            x̂ = denormalize_batch(x̂_norm, normalizer)
            x_target = denormalize_batch(x_target, normalizer)
        else
            x = snapshots[:, :, :, s:s]            # (H,W,C,1)
            x_in = cat(x, μ₀; dims=3) 
            x̂ = reconstruct(enc, dec, x)
        end

        for ch in 1:C
            mat_in = x_target[:, :, ch]
            mat_out = x̂[:, :, ch]


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

            push!(plots, img_in)
            push!(plots, img_out)
        end
    end
    # Global layout spacing controls (tight!)
    p = plot(plots...;
             layout=(nn*C, 2),
             link=:none, legend=false,
             size=(500, 750),
             dpi=200, grid=false)
    
    return p
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    checkpoint = "/home/matth/Thesis/data/models/2025-11-04_09-53-44/checkpoint.jld2"
    # visualize_reconstructions(checkpoint)

end


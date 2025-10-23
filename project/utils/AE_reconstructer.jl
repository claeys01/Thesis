using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/AE_core.jl")
includet("../custom.jl")



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
    # load from checkpoint if a path was given
    if checkpoint_path !== nothing
        checkpoint = JLD2.load(checkpoint_path)
        encoder_state = checkpoint["encoder"]
        decoder_state = checkpoint["decoder"]
        args = Args(; checkpoint["args"]...)

        enc = Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv)
        dec = Decoder(args.input_dim, args.latent_dim; C_next=args.C_conv)
        Flux.loadmodel!(enc, encoder_state)
        Flux.loadmodel!(dec, decoder_state)
    else
        # no checkpoint: require provided encoder, decoder and args
        if encoder === nothing || decoder === nothing
            error("Either checkpoint_path or both encoder and decoder must be provided.")
        end
        if args === nothing
            error("args must be provided when not loading from a checkpoint (needed for data_path, downsample, seed, ...).")
        end
        enc = encoder
        dec = decoder
    end

    # optional seeding
    args.seed > 0 && Random.seed!(args.seed)

    @load args.data_path RHS_data
    snapshots, ids = get_random_snapshots(args.data_path; n=args.n_reconstruct, downsample=args.downsample)

    _, _, C, nn = size(snapshots)
    @info "Selected snapshot indices for reconstruction: $ids"

    # prepare plotting grid: each sample has C rows; two columns (input, recon)
    plots = []
    dirs = ["x" , "y"]
    for s in 1:nn
        x = snapshots[:, :, :, s:s]            # (H,W,C,1)
        x̂ = reconstruct(enc, dec, x)

        for ch in 1:C
            mat_in = x[:, :, ch, 1]
            mat_out = x̂[:, :, ch, 1]

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
    # visualize_reconstructions("/home/matth/Thesis/data/models/2025-10-21_15-28-57")
end
# visualize_reconstructions("/home/matth/Thesis/data/models/2025-10-21_11-43-40/checkpoint.jld2")
using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/AE_core.jl")
includet("../custom.jl")



function visualize_reconstructions(checkpoint_path; n::Int=2, device=Flux.get_device("CPU"), savepath=nothing)
    # load checkpoint (expects keys "encoder", "decoder", "args")
    checkpoint = JLD2.load(checkpoint_path)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    args = Args(; checkpoint["args"]...)

    enc = Encoder(args.input_dim, args.latent_dim)
    dec = Decoder(args.input_dim, args.latent_dim)
    Flux.loadmodel!(enc, encoder_state)
    Flux.loadmodel!(dec, decoder_state)

    args.seed > 0 && Random.seed!(args.seed)

    @load args.data_path RHS_data
    snapshots, ids = get_random_snapshots(args.data_path;n=n, downsample=args.downsample)

    _, _, C, nn = size(snapshots)
    @info "Selected snapshot indices for reconstruction: $ids"

    # prepare plotting grid: each sample has C rows; two columns (input, recon)
    total_rows = nn * C
    # p = plot(layout=(total_rows, 2), size=(500, 750))
    plots = []
    dirs = ["x" , "y"]
    for s in 1:nn
        # println(s)
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
using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/conv_AE_zoo_cpu.jl")  # to get Encoder, Decoder, get_data, reconstruct, Args
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
    println(size(snapshots))
    @info "Selected snapshot indices: $ids"

    # prepare plotting grid: each sample has C rows; two columns (input, recon)
    total_rows = nn * C
    # p = plot(layout=(total_rows, 2), size=(500, 750))
    plots = []
    for s in 1:nn
        # println(s)
        x = snapshots[:, :, :, s:s]            # (H,W,C,1)
        x̂ = reconstruct(enc, dec, x)
        for ch in 1:C
            row = (s-1)*C + ch
            println(row)
            mat_in = x[:, :, ch, 1]
            mat_out = x̂[:, :, ch, 1]
            # println(size(mat_in), size(mat_out))

            μ = mean(mat_in)
            σ = std(mat_in)
            clim = (μ - σ, μ + σ)

            # left plot (input): no colorbar
            img_in = flood(mat_in;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal)

            # right plot (reconstructed): only here show colorbar
            img_out = flood(mat_out;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal)

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
    if savepath !== nothing
        try
            mkpath(dirname(savepath))
        catch
            # ignore if dirname fails or path already exists
        end
        savefig(p, savepath)
    else
        # display(p)
    end
    return p
end

visualize_reconstructions("/home/matth/Thesis/data/saved_models/first/checkpoint.jld2")
using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/conv_AE_zoo_cpu.jl")  # to get Encoder, Decoder, get_data, reconstruct, Args
includet("../custom.jl")



function visualize_reconstructions(checkpoint_path; n::Int=2, seed::Int=42, device=Flux.get_device("CPU"), savepath=nothing)
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

    H, W, C, nn = size(snapshots)
    @info "Selected snapshot indices: $ids"

    # prepare plotting grid: each sample has C rows; two columns (input, recon)
    total_rows = nn * C
    p = plot(layout=(total_rows, 2), size=(500, 750))

    for s in 1:nn
        # keep batch dim so reconstruct accepts input (H,W,C,1)
        x = snapshots[:, :, :, s:s]            # (H,W,C,1)
        x_dev = x
        x̂ = reconstruct(enc, dec, x_dev) # (H,W,C,1)
        x_cpu = cpu(x)

        for ch in 1:C
            row = (s-1)*C + ch
            mat_in = dropdims(x_cpu[:, :, ch, 1], dims=())   # (H,W)
            mat_out = dropdims(x̂[:, :, ch, 1], dims=())

            μ = mean(mat_in)
            σ = std(mat_in)
            clim = (μ - σ, μ + σ)

            img_in  = flood(mat_in,  border=:none, clims=clim)
            img_out = flood(mat_out, border=:none, clims=clim)

            plot!(p, img_in,  subplot=(row, 1), title="s$(ids[s]) ch$(ch) input")
            plot!(p, img_out, subplot=(row, 2), title="s$(ids[s]) ch$(ch) recon")
        end
    end
    if savepath !== nothing
        try
            mkpath(dirname(savepath))
        catch
            # ignore if dirname fails or path already exists
        end
        savefig(p, savepath)
    else
        display(p)
    end
    return p
end

visualize_reconstructions("/home/matth/Thesis/data/saved_models/first/checkpoint.jld2")
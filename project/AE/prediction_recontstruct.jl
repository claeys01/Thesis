using JLD2
using Random
using Plots
using WaterLily

includet("../AE/AE_core.jl")
includet("../utils/AE_normalizer.jl")
includet("../custom.jl")




function period_reconstruct(prediction_path, AE_path)
    checkpoint = JLD2.load(AE_path)
    decoder_state = checkpoint["decoder"]
    normalizer = checkpoint["normalizer"]
    args = Args(; checkpoint["args"]...)
    dec = Decoder(args.output_dim, args.latent_dim; hidden_dim=args.hidden_dim, C_next=args.C_conv, verbose=false)
    Flux.loadmodel!(dec, decoder_state)

    prediction_data = JLD2.load(prediction_path)
    period_pred = prediction_data["period_pred"]
    pred_idx = prediction_data["pred_idx"]

    decoded_pred = denormalize_batch(dec(period_pred), normalizer)[:, :, :, 1]
    @show size(decoded_pred)
    # optional seeding
    args.seed > 0 && Random.seed!(args.seed)

    @load "/home/matth/Thesis/data/datasets/U_128.jld2" data
    preprocess_data!(data; n_samples=-1, clip_bc=args.clip_bc,verbose=false)

    u = data["u"][:, :, :, pred_idx]
    decoded_pred .*= data["μ₀"][:, :, :, pred_idx]

    # ---------- new: contour plots for original u and decoded prediction ----------
    # Expecting arrays of shape (Nx, Ny, C). Plot each channel in a row:
    nx, ny, nc = size(u)
    plots = []
    for ch in 1:nc
        mat_u = Array(u[:, :, ch])           # original
        mat_pred = Array(decoded_pred[:, :, ch])  # reconstructed

        # use mean±std for color limits (consistent between original and pred)
        μ = mean(mat_u)
        σ = std(mat_u)
        clim = (μ - σ, μ + σ)

        p_orig = flood(mat_u;
                       clims=clim,
                       aspect_ratio=:equal,
                       border=:none,
                       framestyle=:none,
                       axis=nothing,
                       ticks=false,
                       colorbar=false,
                       title="orig: channel $ch",
                       titlefontsize=10,
                       legend=false)

        p_pred = flood(mat_pred;
                       clims=clim,
                       aspect_ratio=:equal,
                       border=:none,
                       framestyle=:none,
                       axis=nothing,
                       ticks=false,
                       colorbar=false,
                       title="decoded pred: channel $ch",
                       titlefontsize=10,
                       legend=false)

        push!(plots, p_orig)
        push!(plots, p_pred)
    end

    p = plot(plots...; layout=(nc, 2), size=(500, 250*nc), link=:none)
    display(p)
    savefig(p, "figs/node_predictions/period_prediction_decoded.png")
    # ------------------------------------------------------------------------------

    nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    path="data/latent_data/period_predictions/period_pred.jld2"
    AE_path= "data/saved_models/u/256h_16l/u_100period_100e_4096n_256h_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    period_reconstruct(path, AE_path)
end

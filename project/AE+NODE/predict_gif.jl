using JLD2
using Revise
using LinearAlgebra
using WaterLily
using Plots  # import Plots
using Statistics  # for mean/std
using Printf      # for @sprintf

includet("../NODE/NODE_core.jl")
includet("../NODE/NODE_RE2500_extrapolate.jl")
includet("../AE/Lux_AE.jl")


function make_test_gif(AE_path::String, NODE_path::String)
    (_, dec, _, ps, st, ae_args) = load_trained_AE(AE_path; return_params=true)

    normalizer = load_normalizer(AE_path)

    # load NODE
    node, node_args = load_node(NODE_path)

    # load data
    #   load AE data
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata)


    N = size(simdata.u, 4)
    # @show N
    train_idx = findall(t -> t < ae_args.t_training, simdata.time)
    test_idx = collect(last(train_idx)+1:N)
    idx = collect(1:N)

    downsample_idx = downsample_equal(idx, 500)

    #   load node data
    # _, t_train, _, z0_train = get_NODE_data(node_args.train_latent_path; downsample=-1, verbose=false)
    # _, t_test, _, z0_test = get_NODE_data(node_args.test_latent_path; downsample=-1, verbose=false)
    _, t_total, _, z0_total = get_NODE_data(node_args.total_latent_path; downsample=-1, verbose=false)

    # ẑ_train, ẑ_test = predict_array(node, z0_train; t=t_train), predict_array(node, z0_test; t=t_test)
    ẑ_total = predict_array(node, z0_total; t=t_total)

    û_total, _ = dec(ẑ_total, ps.decoder, st.decoder)
    ẑ_total = nothing
    û_total = denormalize_batch(û_total, normalizer)

    # create plots starting from the same z0, so the neural ode predicts the whole simulation
    plts = []
    dirs = ["x", "y"]
    anim = Plots.Animation()
    # per-channel stats with singleton dims removed: size -> (C,)
    μ = dropdims(mean(simdata.u; dims=(1, 2, 4)), dims=(1, 2, 4))
    σ = dropdims(std(simdata.u;  dims=(1, 2, 4)), dims=(1, 2, 4))
    clims_per_channel = [(μ[ch] - σ[ch], μ[ch] + σ[ch]) for ch in eachindex(μ)]
    
    for (i, idx) in enumerate(downsample_idx)
        panels = []  # collect per-channel plots for this frame
        errs = Float64[]  # per-channel MAE for title

        for ch in 1:2
            u_in = simdata.u[:, :, ch, idx]
            û_out = û_total[:, :, ch, idx] .* simdata.μ₀[:, :, ch, idx]

            diff = abs.(u_in .- û_out)

            plot_in = flood(u_in;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clims_per_channel[ch],
                aspect_ratio=:equal, titlefontsize=8)

            plot_out = flood(û_out;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clims_per_channel[ch],
                aspect_ratio=:equal, titlefontsize=8)

            # viridis contour plot of the difference, rotated CCW, no contour lines
            maxabs = max(abs(minimum(diff)), abs(maximum(diff)))
            diff_rot = rotl90(diff)
            plot_err = contourf(diff_rot;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=(-maxabs, maxabs),
                aspect_ratio=:equal, titlefontsize=8,
                color=Plots.cgrad(:viridis),
                linealpha=0,  # hide contour lines
                linewidth=0)

            # include all three panels per channel
            push!(panels, plot_in, plot_out, plot_err)

            push!(errs, mean(abs.(u_in .- û_out)))
        end

        # title with time and average MAE across channels
        t_now = simdata.time[idx]
        mae = mean(errs)
        ttl = @sprintf("t = %.3f | MAE = %.4e", t_now, mae)

        p = plot(panels...;
             layout=(2, 3),  # 2 channels × (in, out, diff)
             link=:none, legend=false,
             size=(1200, 800),
             dpi=200, grid=false,
             title=ttl, titlefontsize=12)

        frame(anim, p)
        @info "$i / $(length(downsample_idx))"
    end
    gif(anim, "test.gif"; fps=15, show_msg=false)
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"

    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    make_test_gif(AE_path, node_path)
end
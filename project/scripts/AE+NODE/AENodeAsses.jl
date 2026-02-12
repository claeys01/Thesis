# using JLD2
# using Revise
# using LinearAlgebra

# includet("../NODE/NODE_core.jl")
# includet("../NODE/NODE_RE2500_extrapolate.jl")
# includet("../AE/Lux_AE.jl")
using Thesis
using Statistics
"""
Compute the single-step prediction error L_SS as defined in Eq. (11):

    L_SS = ⟨ || ψ(ŵ(t+Δt)) − u(t+Δt) ||₂² ⟩

"""
function L_ss(dec, ẑ_next, u_next)
    per_snapshot_error = sum(abs2, û .- u)
    return mean(per_snapshot_error)
end


function AENodeAsses(AE_path::String, NODE_path::String; saveplot=false)
    # 1. load the trained AE and NODE
    # 2. reconstruct the ẑ with the decoder ψ(ẑ(t+Δt)) of train and test range
    #   2.1 first with the downsamples data and then later check with full dataset
    # 3. 

    # load AE
    (_, dec, _, ps, st, ae_args)  = load_trained_AE(AE_path; return_params=true)

    normalizer = load_normalizer(AE_path)

    # load NODE
    node, node_args = load_node(NODE_path)

    # load data
    #   load AE data
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata)
    N = size(simdata.u, 4)
    @show N
    train_idx = findall(t -> t < ae_args.t_training, simdata.time)
    test_idx = collect(last(train_idx)+1 : N)
    idx = collect(1:N)

    # u_train, u_test = simdata.u[train_idx], simdata.u[test_idx]
    #   load node data
    _, t_train, _, z0_train = Thesis.get_NODE_data(node_args.train_latent_path; downsample=-1, verbose=false)
    _, t_test,  _, z0_test  = Thesis.get_NODE_data(node_args.test_latent_path;  downsample=-1,  verbose=false)    
    z_total, t_total, tspan_total, z0_total = Thesis.get_NODE_data(node_args.total_latent_path; downsample=-1, verbose=false)

    # ẑ_train, ẑ_test = predict_array(node, z0_train; t=t_train), predict_array(node, z0_test; t=t_test)
    @show size(z_total)
    # reconstruct latent prediction
    combi_per_snapshot_error_arr = []
    combi_train_per_snapshot_error_arr = []
    combi_test_per_snapshot_error_arr = []

    AE_per_snapshot_error_arr = []
    AE_train_per_snapshot_error_arr = []
    AE_test_per_snapshot_error_arr = []
    t_down = []

    Δt_pred = 40
    downsample_idx = Thesis.downsample_equal(idx, 500)

    downsample_idx[end] -= Δt_pred

    for i in downsample_idx[1:end-10]
        # single step prediction
        i_pred = i+Δt_pred
        push!(t_down, t_total[i_pred])
        ẑ_next = predict_array(node, z_total[:, i]; t=t_total[i:i_pred])
        û_next, _ = dec(ẑ_next[:,end], ps.decoder, st.decoder)
        û_next = dropdims(û_next, dims=4) .* simdata.μ₀[:, :, :, i_pred]
        u_next, _ = normalize_batch(simdata.u[:, :, :, i_pred]; normalizer = normalizer)
        combi_per_snapshot_error = mean(abs, (u_next .- û_next))
        push!(combi_per_snapshot_error_arr, combi_per_snapshot_error)


        z_next = z_total[:, i_pred]
        u_next_dec, _ = dec(z_next, ps.decoder, st.decoder)
        AE_per_snapshot_error = mean(abs, (u_next .- u_next_dec))
        push!(AE_per_snapshot_error_arr,  AE_per_snapshot_error)

        @info "$(i_pred), $(t_total[i_pred]): $combi_per_snapshot_error"

        if t_total[i] ≤ t_train[end]
            push!(combi_train_per_snapshot_error_arr, combi_per_snapshot_error)  
            push!(AE_train_per_snapshot_error_arr,  AE_per_snapshot_error)
        else
            push!(combi_test_per_snapshot_error_arr, combi_per_snapshot_error)
            push!(AE_test_per_snapshot_error_arr,  AE_per_snapshot_error)

        end
    end
    # @show mean(per_snapshot_error_arr)
    plt = plot(
        t_down,
        combi_per_snapshot_error_arr;
        label = "$Δt_pred step prediction MAE",
        lw = 1.5,
        color = :red,
        xlabel = "Time",
        ylabel = "Mean absolute reconstruction error",
        title = "$Δt_pred step AE–NODE prediction error",
        legend = :topright,
        grid = :on,
        framestyle = :box,
    )
    
    plot!(
        plt,
        t_down,
        AE_per_snapshot_error_arr;
        label = "AE MAE (ground truth)",
        lw = 1.5,
        color = :black,
        xlabel = "Time",
        ylabel = "Mean absolute reconstruction error",
        title = "$Δt_pred step AE–NODE prediction error",
        legend = :topright,
        grid = :on,
        framestyle = :box,
    )

    # optional: horizontal mean line
    plot!(
        plt,
        [0, t_train[end]],
        [mean(combi_train_per_snapshot_error_arr), mean(combi_train_per_snapshot_error_arr)];
        color = :red,
        lw = 1.5,
        linestyle = :dash,
        label = "Mean region error",
    )
    plot!(
        plt,
        [t_test[1], t_test[end]],
        [mean(combi_test_per_snapshot_error_arr), mean(combi_test_per_snapshot_error_arr)];
        color = :red,
        lw = 1.5,
        linestyle = :dash,
        label = "",
    )

    plot!(
        plt,
        [0, t_train[end]],
        [mean(AE_train_per_snapshot_error_arr), mean(AE_train_per_snapshot_error_arr)];
        color = :black,
        lw = 1.5,
        linestyle = :dash,
        label = ""
    )

    plot!(
        plt,
        [t_test[1], t_test[end]],
        [mean(AE_test_per_snapshot_error_arr), mean(AE_test_per_snapshot_error_arr)];
        color = :black,
        lw = 1.5,
        linestyle = :dash,
        label = "",
    )

    Thesis.region_spans!(plt, t_train, t_test)

    if saveplot
        node_dir = split(node_path, "/")
        node_dir = join(node_dir[1:end-1],"/")
        figpath = joinpath(node_dir, "single_step_err.png")
        @info "plot saved to $figpath"
        savefig(plt, figpath)
    end
    return plt
end


if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
    AENodeAsses(AE_path, node_path; saveplot=true)

end
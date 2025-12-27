using JLD2
using Revise
using LinearAlgebra

includet("../NODE/NODE_core.jl")
includet("../NODE/NODE_RE2500_extrapolate.jl")
includet("../AE/Lux_AE.jl")


"""
Compute the single-step prediction error L_SS as defined in Eq. (11):

    L_SS = ⟨ || ψ(ŵ(t+Δt)) − u(t+Δt) ||₂² ⟩

"""
function L_ss(dec, ẑ_next, u_next)
    per_snapshot_error = sum(abs2, û .- u)
    return mean(per_snapshot_error)
end



function AENodeAsses(AE_path::String, NODE_path::String)
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

    downsample_idx = downsample_equal(idx, 500)

    # u_train, u_test = simdata.u[train_idx], simdata.u[test_idx]


    #   load node data
    _, t_train, _, z0_train = get_NODE_data(node_args.train_latent_path; downsample=-1, verbose=false)
    _, t_test,  _, z0_test  = get_NODE_data(node_args.test_latent_path;  downsample=-1,  verbose=false)    
    z_total, t_total, tspan_total, z0_total = get_NODE_data(node_args.total_latent_path; downsample=-1, verbose=false)

    # ẑ_train, ẑ_test = predict_array(node, z0_train; t=t_train), predict_array(node, z0_test; t=t_test)
    @show size(z_total)
    # reconstruct latent prediction

    per_snapshot_error_arr = []
    t_down = []
    for i in downsample_idx
        # single step prediction
        push!(t_down, t_total[i+1])
        ẑ_next = predict_array(node, z_total[:, i]; t=t_total[i:i+1])
        û_next, _ = dec(ẑ_next[:,end], ps.decoder, st.decoder)
        û_next = dropdims(û_next, dims=4)
        u_next, _ = normalize_batch(simdata.u[:, :, :, i+1]; normalizer = normalizer)
        per_snapshot_error = mean(abs, (u_next .- û_next))

        # rollout error

        @info "$(i+1), $(t_total[i+1]): $per_snapshot_error"
        push!(per_snapshot_error_arr, per_snapshot_error)        

        # @show z_total[:, i] ẑ_next
    end
    @show mean(per_snapshot_error_arr)
    plt = plot(t_down, per_snapshot_error_arr)
    region_spans!(plt, t_train, t_test)
    display(plt)


end


if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"

    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    AENodeAsses(AE_path, node_path)
end
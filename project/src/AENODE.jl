mutable struct AENODE{P,S}
    encoder::Encoder
    decoder::Decoder
    normalizer::Normalizer
    ae_args::LuxArgs
    NODE::NODE
    node_args::NodeArgs
    knn_ood::KNNOOD
    ae_params::P
    ae_state::S
end

function AENODE(AE_path::String, NODE_path::String; verbose=false, k=5, q=0.99)
    enc, dec, _, ps, st, ae_args = load_trained_AE(AE_path; return_params=true)
    normalizer = load_normalizer(AE_path)
    node, node_args = load_node(NODE_path; verbose=verbose)
    knnood = fit_knn_ood(get_NODE_data(node_args.train_latent_path; downsample=node_args.downsample, verbose=verbose)[1], k=k, q=q)
    return AENODE(enc, dec, normalizer, ae_args, node, node_args, knnood,
        ps,  # concrete NamedTuple type inferred
        st   # concrete NamedTuple type inferred
    )
end

# a function that predicts the new flow field of a simulation for a selected amount of timesteps, 
function predict_n(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray, nₜ::Int64, t₀::Float32; 
                Δt::Float32=0.35f0, 
                return_traj::Bool=false)
    @assert size(u) == size(μ₀) "u and μ₀ must be the same size"
    if !ispow2(size(u, 1)) || !ispow2(size(μ₀, 1))
        u, μ₀ = remove_ghosts(u), remove_ghosts(μ₀)
    end
    u, _ = @timeit to "normalize batch" normalize_batch(u; normalizer=aenode.normalizer)
    tmp = cat(u, μ₀; dims=3)                      # (H, W, C)
    u_in = reshape(tmp, size(tmp,1), size(tmp,2), size(tmp,3), 1)  # (H, W, C, 1)
    
    # compress simulation flow field to latent space
    z, _ = @timeit to "encode" aenode.encoder(u_in, aenode.ae_params.encoder, aenode.ae_state.encoder)
    z = vec(z)

    # predict in latent space using node
    t = return_traj ? range(t₀, step=Δt/32.0f0, length=nₜ+1) : range(t₀, step=nₜ * Δt/32.0f0, length=2)
    # 32 is the characteristic length of the simulation, need to take out hard coding later and pass it to ae or node args
    ẑ = @timeit to "NODE integrate" predict_array(aenode.NODE, z; t=t)
    # decompress latent prediction
    û, _ = @timeit to "decode" aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    û = @timeit to "denormalize" denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, length(t))
    û = û[:,:,:,2:end]
    # if desired, return trajectory of flow fields, or return end of trajectory as a simulation object
    return_traj ? (return û) : (return û[:, :, :, end])
end

function predict_n!(sim::BiotSimulation, aenode::AENODE, nₜ::Int64; 
    Δt::Float32=0.35f0, impose_biot=false)
    û = predict_n(aenode, sim.flow.u, sim.flow.μ₀, nₜ, Float32(sim_time(sim));
        Δt=Δt, return_traj=false)
    
    @timeit to "insert pred" insert_prediction!(sim, û) # insert predicted flow field into sim object
    Δt_arr = [Δt for _ in 1:nₜ]
    append!(sim.flow.Δt, Δt_arr)
    push!(sim.flow.Δt, WaterLily.CFL(sim.flow))
    @timeit to "impose biot" impose_biot_bc!(sim) #  update pressure
end

function predict_flex(aenode::AENODE, sim::BiotSimulation; Δt::Float32=0.35f0, impose_biot=false)
    û, n_integr = predict_flex(aenode, sim.flow.u, sim.flow.μ₀, Float32(sim_time(sim)); Δt=Δt)
    if isnothing(û)
        return sim, n_integr
    end
    insert_prediction!(sim, û)
    Δt_arr = [Δt for _ in 1:n_integr]
    append!(sim.flow.Δt, Δt_arr)
    push!(sim.flow.Δt, WaterLily.CFL(sim.flow))
    impose_biot && @timeit to "impose biot" impose_biot_bc!(sim)
    return sim, n_integr
end

function predict_flex(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray, t₀::Float32; Δt::Float32=0.35f0 )
    @assert size(u) == size(μ₀) "u and μ₀ must be the same size"
    if !ispow2(size(u, 1)) || !ispow2(size(μ₀, 1))
        u, μ₀ = remove_ghosts(u), remove_ghosts(μ₀)
    end
    u, _ = @timeit to "normalize batch" normalize_batch(u; normalizer=aenode.normalizer)
    tmp = cat(u, μ₀; dims=3)                      # (H, W, C)
    u_in = reshape(tmp, size(tmp,1), size(tmp,2), size(tmp,3), 1)  # (H, W, C, 1)
    
    # compress simulation flow field to latent space
    z, _ = @timeit to "encode" aenode.encoder(u_in, aenode.ae_params.encoder, aenode.ae_state.encoder)
    z = vec(z)
    if KNN_score(aenode.knn_ood, z) > aenode.knn_ood.threshold
        @warn "Encodeded flow not similar to training data, NODE integration can be wrong"
        return nothing, 0
    end

    # NODE integration untill cutoff criteria is met.
    tₙ = t₀ + Δt/32.0f0
    n_integr = 0
    ẑ = Thesis.predict_array(aenode.NODE,  z; t=[t₀, tₙ], onlysol=true)[:, end]
    while true 
        n_integr += 1
        knn_score = KNN_score(aenode.knn_ood, ẑ)
        if knn_score > aenode.knn_ood.threshold
            @warn "NODE integration too far outside of training distances, cutting of integration after $n_integr steps"
            break
        else
            tₙ += Δt/32.0f0
            ẑ = Thesis.predict_array(aenode.NODE,  z; t=[t₀, tₙ], onlysol=true)[:, end]
        end
    end
    # decode latent prediction
    û, _ = @timeit to "decode" aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    û = @timeit to "denormalize" denormalize_batch(û, aenode.normalizer) .* μ₀
    return û[:, :, :, end], n_integr
end

# node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"

# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

# aenode = AENODE(AE_path, node_path)
# sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
# predict_n(aenode, sim, 4)
# nothing


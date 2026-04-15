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
    ae_bundle, ae_args = load_trained_AE(AE_path; return_params=true)
    normalizer = load_normalizer(AE_path)
    node, node_args = load_node(NODE_path; verbose=verbose)
    # knnood = fit_knn_ood(get_NODE_data(node_args.train_latent_path; downsample=node_args.downsample, verbose=verbose)[1], k=k, q=q)
    # knnood = fit_knn_ood(get_latent_vectors(ae, ps, st, normalizer, ae_args; downsample=node_args.downsample)[1])
    # return AENODE(ae.encoder, ae.decoder, normalizer, ae_args, node, node_args, knnood,
        # ps,  # concrete NamedTuple type inferred
        # st   # concrete NamedTuple type inferred
    # )
    return AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=verbose, k=k, q=q)
end

function AENODE(ae_bundle, node::NODE, ae_args::LuxArgs, node_args::NodeArgs, normalizer::Normalizer; verbose=false, k=5, q=0.99)
    knnood = fit_knn_ood(get_latent_vectors(ae_bundle, normalizer, ae_args; downsample=node_args.downsample)[1])
    verbose && @info "AENODE Initialized"
    return AENODE(ae_bundle.ae.encoder, ae_bundle.ae.decoder, normalizer, ae_args, node, node_args, knnood,
        ae_bundle.ps,  # concrete NamedTuple type inferred
        ae_bundle.st   # concrete NamedTuple type inferred
    )
end


function encode_flow(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray)
    @assert size(u) == size(μ₀) "u and μ₀ must be the same size"
    if !ispow2(size(u, 1)) || !ispow2(size(μ₀, 1))
        u, μ₀ = remove_ghosts(u), remove_ghosts(μ₀)
    end
    u, _ = @timeit to "normalize batch" normalize_batch(u; normalizer=aenode.normalizer)
    tmp = cat(u, μ₀; dims=3)                      # (H, W, C)
    u_in = reshape(tmp, size(tmp,1), size(tmp,2), size(tmp,3), 1)  # (H, W, C, 1)
    z, _ = @timeit to "encode" aenode.encoder(u_in, aenode.ae_params.encoder, aenode.ae_state.encoder)
    return vec(z), μ₀
end

function decode_flow(aenode::AENODE, ẑ, μ₀)
    û, _ = @timeit to "decode" aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    û = @timeit to "denormalize" denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, size(ẑ, 2))
    return size(û, 4) == 1 ? dropdims(û; dims=4) : û[:,:,:,2:end]
end

function apply_prediction!(sim::BiotSimulation, û, Δt::Float32, n_steps::Int; impose_biot=false)
    @timeit to "insert pred" insert_prediction!(sim, û)
    sim.flow.u⁰[2:end-1, 2:end-1, :] .= û
    fill!(sim.flow.p, 0)
    append!(sim.flow.Δt, Δt * (n_steps - 1))
    push!(sim.flow.Δt, WaterLily.CFL(sim.flow))
    impose_biot && @timeit to "impose biot" impose_biot_bc!(sim)
end

# a function that predicts the new flow field of a simulation for a selected amount of timesteps, 
function predict_n(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray, nₜ::Int64, t₀::Float32; 
                Δt::Float32=0.35f0, 
                return_traj::Bool=false)
    z, μ₀ = encode_flow(aenode, u, μ₀)

    # predict in latent space using node
    t = return_traj ? range(t₀, step=Δt/32.0f0, length=nₜ+1) : range(t₀, step=nₜ * Δt/32.0f0, length=2)
    # 32 is the characteristic length of the simulation, need to take out hard coding later and pass it to ae or node args
    ẑ = @timeit to "NODE integrate" predict_array(aenode.NODE, z; t=t)
    # decompress latent prediction
    # û, _ = @timeit to "decode" aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    # û = @timeit to "denormalize" denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, length(t))
    # û = û[:,:,:,2:end]
    decode_flow(aenode, ẑ, μ₀)
    # if desired, return trajectory of flow fields, or return end of trajectory as a simulation object
    return_traj ? (return û) : (return û[:, :, :, end])
end

function predict_n!(sim::BiotSimulation, aenode::AENODE, nₜ::Int64; 
    Δt::Float32=0.35f0, impose_biot=false)
    û = predict_n(aenode, sim.flow.u, sim.flow.μ₀, nₜ, Float32(sim_time(sim));
        Δt=Δt, return_traj=false)
    apply_prediction!(sim, û, Δt, nₜ; impose_biot=impose_biot)
end

function predict_flex(aenode::AENODE, sim::BiotSimulation; Δt::Float32=0.35f0, impose_biot=false, next_save=0.25)
    û, n_integr, retrain_required = predict_flex(
        aenode,
        sim.flow.u,
        sim.flow.μ₀,
        Float32(sim_time(sim));
        Δt=Δt,
        next_save=next_save,
    )
    if isnothing(û)
        return sim, n_integr, retrain_required
    end
    apply_prediction!(sim, û, Δt, n_integr; impose_biot=impose_biot)
    return sim, n_integr, retrain_required
end

function predict_flex(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray, t₀::Float32; Δt::Float32=0.35f0, next_save=0.25)
    z, μ₀ = encode_flow(aenode, u, μ₀)
    retrain_required = false
    z_score = KNN_score(aenode.knn_ood, z)
    if z_score > aenode.knn_ood.threshold
        @warn "Encoded flow not similar to training data, AE and NODE should be retrained" z_score threshold=aenode.knn_ood.threshold
        return nothing, 0, true
    end

    # NODE integration untill cutoff criteria is met.
    tₙ = t₀ + Δt/32.0f0
    n_integr = 1
    ẑ = predict_array(aenode.NODE,  z; t=[t₀, tₙ], onlysol=true)[:, end]
    while true 
        knn_score = KNN_score(aenode.knn_ood, ẑ)
        if knn_score > aenode.knn_ood.threshold
            @warn "NODE integration too far outside of training distances, cutting of integration after $n_integr steps" z_score threshold=aenode.knn_ood.threshold
            retrain_required = true
            break
        elseif tₙ ≥ next_save
            @warn "NODE integration exceeds time for saving mean flow statistics, cutting of integration at $tₙ "
            break
        else
            tₙ += Δt/32.0f0
            ẑ = predict_array(aenode.NODE,  z; t=[t₀, tₙ], onlysol=true)[:, end]
            n_integr += 1
        end
    end
    û = decode_flow(aenode, ẑ[:, end], μ₀)
    return û, n_integr, retrain_required
end


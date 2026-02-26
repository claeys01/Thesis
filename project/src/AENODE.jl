mutable struct AENODE{P,S}
    encoder::Encoder
    decoder::Decoder
    normalizer::Normalizer
    ae_args::LuxArgs
    NODE::NODE
    node_args::NodeArgs
    ae_params::P
    ae_state::S
end

function AENODE(AE_path::String, NODE_path::String; verbose=false)
    enc, dec, _, ps, st, ae_args = load_trained_AE(AE_path; return_params=true)
    normalizer = load_normalizer(AE_path)
    node, node_args = load_node(NODE_path; verbose=verbose)
    return AENODE(enc, dec, normalizer, ae_args, node, node_args,
        ps,  # concrete NamedTuple type inferred
        st   # concrete NamedTuple type inferred
    )
end

# a function that predicts the new flow field of a simulation for a selected amount of timesteps, 
function predict_n(aenode::AENODE, u::AbstractArray, μ₀::AbstractArray, nₜ::Int64, t₀::Float32; 
                Δt::Float32=0.35f0, 
                return_traj::Bool=false, 
                impose_biot::Bool=false)
    @assert size(u) == size(μ₀) "u and μ₀ must be the same size"
    if !ispow2(size(u, 1)) || !ispow2(size(μ₀, 1))
        u, μ₀ = remove_ghosts(u), remove_ghosts(μ₀)
    end
    u, _ = normalize_batch(u; normalizer=aenode.normalizer)
    tmp = cat(u, μ₀; dims=3)                      # (H, W, C)
    u_in = reshape(tmp, size(tmp,1), size(tmp,2), size(tmp,3), 1)  # (H, W, C, 1)
    
    # compress simulation flow field to latent space
    z, _ = aenode.encoder(u_in, aenode.ae_params.encoder, aenode.ae_state.encoder)
    z = vec(z)

    # predict in latent space using node
    t = return_traj ? range(t₀, step=Δt/32.0f0, length=nₜ+1) : range(t₀, step=nₜ * Δt/32.0f0, length=2)
    # 32 is the characteristic length of the simulation, need to take out hard coding later and pass it to ae or node args
    ẑ = predict_array(aenode.NODE, z; t=t)

    # decompress latent prediction
    û, _ = aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    û = denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, length(t))
    û = û[:,:,:,2:end]
    # if desired, return trajectory of flow fields, or return end of trajectory as a simulation object
    if impose_biot
        return_traj ? (return impose_biot_bc_on_snapshot(û)) : (return impose_biot_bc_on_snapshot(û[:, :, :, end]))
    else
        return_traj ? (return û) : (return û[:, :, :, end])
    end
end

function predict_n!(sim::BiotSimulation, aenode::AENODE, nₜ::Int64; 
    Δt::Float32=0.35f0, impose_biot=false)
    û = predict_n(aenode, sim.flow.u, sim.flow.μ₀, nₜ, Float32(sim_time(sim));
        Δt=Δt, return_traj=false, impose_biot=false)
    
    insert_prediction!(sim, û) # insert predicted flow field into sim object
    Δt_arr = [Δt for _ in 1:nₜ]
    append!(sim.flow.Δt, Δt_arr)
    impose_biot_bc!(sim) #  update pressure
end

# node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"

# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

# aenode = AENODE(AE_path, node_path)
# sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
# predict_n(aenode, sim, 4)
# nothing

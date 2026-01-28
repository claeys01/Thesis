using Revise

includet("../NODE/NODE_core.jl")
includet("../AE/Lux_AE.jl")
includet("../simulations/vortex_shedding_biot_savart.jl")
includet("../custom.jl")
includet("../utils/AE_normalizer.jl")



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

function AENODE(AE_path::String, NODE_path::String)
    enc, dec, _, ps, st, ae_args = load_trained_AE(AE_path; return_params=true)
    normalizer = load_normalizer(AE_path)
    node, node_args = load_node(NODE_path)
        return AENODE(enc, dec, normalizer, ae_args, node, node_args,
        ps,  # concrete NamedTuple type inferred
        st   # concrete NamedTuple type inferred
    )
end


# a function that predicts the new flow field of a simulation for a selected amount of timesteps, 
function predict_n(aenode::AENODE, sim::BiotSimulation, nₜ::Int64; Δt::Float32 = Float32(0.25), return_traj::Bool=true)
    # create input data from simulation
    u, μ₀ = remove_ghosts(sim.flow.u), remove_ghosts(sim.flow.μ₀)
    u, _ = normalize_batch(u; normalizer=aenode.normalizer)
    u_in = cat(u, μ₀; dims=3)

    # compress simulation flow field to latent space
    z, _ = aenode.encoder(u_in, aenode.ae_params.encoder, aenode.ae_state.encoder)
    z = vec(z)

    # predict in latent space using node
    # t = range(Float32(Δt), step=Float32(Δt), length=nₜ)  # efficient, no extra alloc
    t = Array(Float32.([sim_time(sim), sim_time(sim) + nₜ*Δt]))
    ẑ = predict_array(aenode.NODE, z; t=t)
    
    # decompress latent prediction
    û, _ = aenode.decoder(ẑ, aenode.ae_params.decoder, aenode.ae_state.decoder)
    # û = denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, nₜ) 
    û = denormalize_batch(û, aenode.normalizer) .* repeat(μ₀, 1, 1, 1, 2) 

    # if desired, return trajectory of flow fields, or return end of trajectory as a simulation object
    return_traj ? (return impose_biot_bc_on_snapshot(û)) : (return impose_biot_bc_on_snapshot(û[:, :, :, end]; return_sim=true))
end

# node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"

# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

# aenode = AENODE(AE_path, node_path)
# sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
# predict_n(aenode, sim, 4)
# nothing

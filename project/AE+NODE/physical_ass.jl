using JLD2
using Revise
using LinearAlgebra
using WaterLily


includet("../NODE/NODE_core.jl")
includet("../NODE/NODE_RE2500_extrapolate.jl")
includet("../AE/Lux_AE.jl")
includet("../custom.jl")


function physical_ass(AE_path::String, NODE_path::String)
    # load AE
    (_, dec, _, ps, st, ae_args)  = load_trained_AE(AE_path; return_params=true)
    normalizer = load_normalizer(AE_path)
    
    # load NODE
    node, node_args = load_node(NODE_path)
    
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata)
    u_0 = copy(simdata.u[:, :, : , 1])
    
    # @show typeof(simdata.u)
    # @show typeof(u_0), eltype(u_0), size(u_0)
    # @show mean(u_0)
    # @show any(isnan, u_0), any(isinf, u_0)

    # simdata = nothing

    s = strain_field(u_0)
    # @show size(s), mean(s), std(s)
    # @show any(isnan, s), any(isinf, s)
    # @show count(isnan, s), count(isinf, s)

    # @show s
    ε = kinetic_energy_dissipation(u_0)
    @show mean(ε)
    GC.gc()
end
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
    physical_ass(AE_path, node_path)
end
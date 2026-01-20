using Random
using WaterLily, BiotSavartBCs
using Plots  # ensure WaterLily's Plots extension (flood) is available

includet("../AE/Lux_AE.jl")
includet("../NODE/NODE_core.jl")
includet("../utils/AE_normalizer.jl")
includet("../custom.jl")
includet("../utils/SimDataTypes.jl")
includet("../simulations/vortex_shedding_biot_savart.jl")
using .SimDataTypes

Random.seed!(42)
node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

(_, dec, _, ps, st, ae_args) = load_trained_AE(AE_path; return_params=true)

normalizer = load_normalizer(AE_path)
node, node_args = load_node(node_path)

_, t_total, _, z0_total = get_NODE_data(node_args.total_latent_path; downsample=10, verbose=false)

ẑ_total = predict_array(node, z0_total; t=t_total)
û_total, _ = dec(ẑ_total, ps.decoder, st.decoder)
û_total = denormalize_batch(û_total, normalizer)
û = û_total[:,:,:,1]

# @show size(û_total)
@info "flow fields predicted"

# defining simulation
n=2^8
Re = 2500
sim_shedding = circle_shedding_biot(mem=Array, Re=Re, n=n, m=n)

@show size(sim_shedding.flow.p)

û .*= sim_shedding.flow.μ₀[2:end-1,2:end-1,:]
plt1 = flood(û[:, :, 1])
û = impose_biot_bc_on_snapshot(û)
@show size(û), size(sim_shedding.flow.μ₀)
plt2 = flood(û[:, :, 1])

divergence = div_field(û)
@show mean(divergence)
# lims = (mean(û) -std(û), mean(û)+std(û))

plt = plot(plt1, plt2)
display(plt)

# ε = kinetic_energy_dissipation(û)

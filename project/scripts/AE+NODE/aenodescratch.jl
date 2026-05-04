using Thesis
using Statistics
using WaterLily

# u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
# sim = circle_shedding_biot(; mem=Array, perturb=false)
# sim.flow.u .= u₀
# sim_step!(sim)

# arr = zeros(size(sim.flow.σ))
# WaterLily.measure_sdf!(arr, sim.body)
# @show size(arr)
# display(flood(arr))
# sim_step!(sim, 5)
# display(flood(sim.flow.u[:, :, 1]))

# root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
# AE_path_tl1 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
# AE_path_tl1 = joinpath(root_path, AE_path_tl1)

# normalizer = load_normalizer(AE_path_tl1)
# ae_bundle, ae_args = load_trained_AE(AE_path_tl1)

# node_path = "data/saved_models/NODE/16/RE2500/TL1_E500_curldiv_MS_Adam_250/node_params.jld2"
# node_path = joinpath(root_path, node_path)
# node, node_args = load_node(node_path)

# aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

# next_save = 0.25 # in ctu
# sim = circle_shedding_biot(; mem=Array, perturb=false)
# sim.flow.u .= u₀
# sim, n_integr, retrain_required = predict_flex(aenode, sim;  impose_biot=false, next_save=next_save)
# @show n_integr

simdata_path = "data/inline_runs/hpc_inline/U_inline.jld2"

simdata = load_simdata(simdata_path)

force_plot = plot(simdata.time, last.(simdata.force))

display(force_plot)



nothing
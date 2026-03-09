using Thesis
using WaterLily

reset_timer!(to::TimerOutput)

node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

reset_timer!(to::TimerOutput)

# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)

random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]
sim = circle_shedding_biot(; mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

pred_Δt = 0.35f0
n_pred = 16
sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
@timeit to "sim_step" sim_step!(sim)
û = @timeit to "predict_n" predict_n!(sim, aenode, n_pred; Δt=pred_Δt, impose_biot=true)

display(to)
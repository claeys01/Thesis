using Thesis
using Thesis: insert_prediction!, predict_flexible
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Plots
using TimerOutputs

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

reset_timer!(to::TimerOutput)

# 1000 with physics in loss func
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)

random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)
sim_meanflow = MeanFlow(sim.flow; uu_stats=true)
cfl = WaterLily.CFL(sim.flow)
@show cfl

train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_end = 50
n_pred = 32
n_switch = 100
pred_Δt = 0.35f0
with_pred = true

û, n_integr = predict_flexible(aenode, u, μ₀, t₀) # predicts 131 steps ahead
insert_prediction!(sim, û)
cfl = WaterLily.CFL(sim.flow)

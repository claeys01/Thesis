using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots


params = InlineParams()

savedir = joinpath("data", "inline_runs", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline.jld2")

u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
sim = circle_shedding_biot(; mem=Array, perturb=false)

root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
AE_path_tl1 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE_path_tl1 = joinpath(root_path, AE_path_tl1)

normalizer = load_normalizer(AE_path_tl1)
ae_bundle, ae_args = load_trained_AE(AE_path_tl1)

node_path = "data/saved_models/NODE/16/RE2500/TL1_E500_curldiv_MS_Adam_250/node_params.jld2"
node_path = joinpath(root_path, node_path)
node, node_args = load_node(node_path)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

hs = HybridState(sim, aenode, params, savedir, AE_path_tl1, node_path)

simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path)
# display(plot_meanflow_comparison(hs.sim_meanflow, hs.ref_meanflow))

run_hybrid!(hs)

if hs.retrain_needed
    GC.gc()
    @info "Retraining triggered at sim_time=$(sim_time(hs.sim)), step=$(hs.step)"
    push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Cutoff"))

    println("continueing to run simulation without AENODE")
    simdata = run_warmup!(hs, sim_time(hs.sim) + 5; simdata=simdata, save_path=simdata_path)

    AE_path_tl2 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL2_E300_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
    AE_path_tl2 = joinpath(root_path, AE_path_tl2)
    normalizer = load_normalizer(AE_path_tl2)
    ae_bundle, ae_args = load_trained_AE(AE_path_tl2)

    node_path_tl2 = "data/saved_models/NODE/16/RE2500/TL2_E300_curldiv_MS_Adam_250/node_params.jld2"
    node_path_tl2 = joinpath(root_path, node_path_tl2)

    node, node_args = load_node(node_path_tl2)

    hs.aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)
    hs.AE_path = AE_path_tl2
    hs.node_path = node_path_tl2
    hs.retrain_needed = false
    hs.step = 0
    
    push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Restarted"))
    run_hybrid!(hs)
end

# simdata = run_warmup!(hs, params.t_accel_end; simdata=simdata, save_path=simdata_path)
@show hs.mode_log

save_results(hs)

# @show size(hs.sim_meanflow.t)
# @show size(hs.ref_meanflow.t)
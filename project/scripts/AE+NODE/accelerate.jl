using Thesis
using Thesis: get_forces
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Plots
using TimerOutputs

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
reset_timer!(to::TimerOutput)

node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)
sim.flow.u .= simdata.u[:, :, :, 1]
sim_step!(sim)
sim_meanflow = MeanFlow(sim.flow; uu_stats=true)

train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_end = 20
n_pred = 32
n_switch = 100
pred_Δt = 0.35f0
save_interval = 0.25
step = 1

res = AccelResults()
ref_sim = deepcopy(sim)
next_save = save_interval
ref_meanflow = MeanFlow(ref_sim.flow; uu_stats=true)

# warmup
predict_n!(deepcopy(sim), aenode, n_pred; Δt=pred_Δt, impose_biot=true)

cID = "biot_sheddding"
WaterLily.logger(cID)

while sim_time(sim) < t_end
    if step % n_switch == 0
        predict_wall_time = @elapsed begin
            predict_n!(sim, aenode, n_pred; Δt=pred_Δt, impose_biot=false)
        end
        # patch pressure from training data
        closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
        sim.flow.p .= simdata.p[:, :, closest_idx]

        sim_dt = n_pred * pred_Δt * sim.U / sim.L
        forces = record_prediction!(res, sim, predict_wall_time, sim_dt, step)
        println(" Inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim), digits=4)), wall time: $(round(predict_wall_time*1000, digits=4)) ms, force: $forces")
    else
        wall_time = @elapsed sim_step!(sim; remeasure=true)
        record_waterlily_step!(res, sim, wall_time)
    end

    while sim_time(sim) > sim_time(ref_sim)
        step_reference!(res, ref_sim)
    end

    if sim_time(sim) > next_save
        WaterLily.update!(sim_meanflow, sim.flow)
        WaterLily.update!(ref_meanflow, ref_sim.flow)
        next_save = sim_time(sim) + save_interval
    end

    step += 1
end

plot_logger("$(cID).log")

print_metrics(res; pred_label="(n=$n_pred fixed)")

plt_combined = plot_accel_combined(res, t_train, t_test, t_end)
rst_comp_plot = plot_rst_comparison(sim_meanflow, ref_meanflow)
plt_meanflow = plot_meanflow_comparison(sim_meanflow, ref_meanflow)

display(plt_combined)
display(rst_comp_plot)
display(plt_meanflow)

savedir = "figs/acceleration/t$(t_end)_np$(n_pred)_ns$(n_switch)/"
save_accel_plots(savedir, plt_combined, rst_comp_plot, plt_meanflow)
using Thesis
using JLD2
using Statistics
using Plots

base_dir = "data/saved_models/inline_runs_hpc/base"
# base_dir = "data/inline_runs/2026-06-05_14-43"

hs_path = joinpath(base_dir, "hybrid_state.jld2")
@load hs_path res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir

# Post-processing only — mirrors save_results(hs) but writes nothing to disk.
print_metrics(res; pred_label="(flexible OOD)",
    avg_steps_per_pred=isempty(n_integrs) ? nothing : mean(n_integrs),
    sim_meanflow=sim_meanflow, ref_meanflow=ref_meanflow, mode_log=mode_log)


ref_drag = first.(res.forces_ref)
ref_lift = last.(res.forces_ref)

hyb_drag = first.(res.hybrid_forces_wat)
hyb_lift = last.(res.hybrid_forces_wat)


meanplot = plot_meanflow_comparison(sim_meanflow, ref_meanflow;
    savedir=joinpath(base_dir, "meanflow_panels"))
display(meanplot)

rst_plot = plot_rst_comparison(sim_meanflow, ref_meanflow)
display(rst_plot)

# sim_u, sim_v = sim_meanflow.U[:, :, 1], sim_meanflow.U[:, :, 2]
# ref_u, ref_v = ref_meanflow.U[:, :, 1], ref_meanflow.U[:, :, 2]

# u_diff = abs.(sim_u .- ref_u)
# v_diff = abs.(sim_v .- ref_v)



# display(plot(res.hybrid_time_wat, hyb_lift))

# plt_combined = plot_accel_combined(res, params.t_accel_end; mode_log=mode_log)

# plt_forces = plot_forces_comparison(res, params.t_accel_end; mode_log=mode_log)
# savefig(plt_forces, joinpath(base_dir, "hybrid_forces.pdf"))
# plt_timing, plt_total = plot_timing_bars(res)
# display(plt_timing); display(plt_total)
# savefig(plt_timing, joinpath(base_dir, "bars_timing.png")); savefig(plt_total, joinpath(base_dir, "bars_total.png"))
# savefig(plt_forces, joinpath(base_dir, "hybrid_forces.pdf"))

# rst_comp_plot = plot_rst_comparison(sim_meanflow, ref_meanflow)
# display(rst_comp_plot)

# plt_meanflow = plot_meanflow_comparison(sim_meanflow, ref_meanflow)
# display(plt_meanflow)

# sim = circle_shedding_biot(; mem=Array, perturb=false)
# @show sim.L
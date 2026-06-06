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


params.sample_interval == 0 ||
    @warn "sample_interval > 0: not every step recorded, warmup wall times will be undercounted" params.sample_interval

is_pred = falses(length(res.hybrid_time_wat))
for r in res.pred_ranges
    is_pred[first(r)+1:last(r)] .= true
end
wl_idx  = findall(!, is_pred)
wl_t    = res.hybrid_time_wat[wl_idx]          # WaterLily-step timestamps
wl_wall = res.hybrid_waterlily_wall_times      # aligned 1:1 with wl_t
@assert length(wl_t) == length(wl_wall)

training_spans = [(l.t_start, l.t_end) for l in mode_log if l.mode == "Training"]
println("\n--- Hybrid-only warmup wall time (sim_step! only) ---")
for (k, (t0, t1)) in enumerate(training_spans)
    sel = k == 1 ? findall(t -> t <= t1, wl_t) : findall(t -> t0 < t <= t1, wl_t)
    s = sum(wl_wall[sel])
    label = k == 1 ? "initial warmup     " : "post-flag warmup #$(k-1)"
    println("  $label  steps=$(rpad(length(sel),5)) wall=$(round(s, digits=2)) s  ($(round(s/60, digits=3)) min)")
end

# Wall time of each hybrid section (the "Hybrid" mode_log spans). Unlike the warmups
# these include both the WaterLily steps (wl_wall) AND the NODE prediction calls
# (res.hybrid_predict_wall_times, one entry per inserted rollout, timestamped at
# res.pred_idx). A small tolerance is added on the upper endpoint: the cutoff
# prediction/step that triggered retraining lands exactly on t_end, but data
# timestamps are rounded to 4 digits while mode_log stores the raw value, so they can
# differ by one rounding unit. The next section's first step is ~0.01 away, well
# outside `tol`, so there is no double-counting.
pred_t    = res.hybrid_time_wat[res.pred_idx]   # one timestamp per inserted prediction
pred_wall = res.hybrid_predict_wall_times       # aligned 1:1 with pred_t
@assert length(pred_t) == length(pred_wall)
tol = 1f-3
hybrid_spans = [(l.t_start, l.t_end) for l in mode_log if l.mode == "Hybrid"]
println("\n--- Hybrid section wall time (CFD steps + NODE predictions) ---")
hybrid_total = 0.0
for (k, (t0, t1)) in enumerate(hybrid_spans)
    cfd  = sum(wl_wall[findall(t -> t0 < t <= t1 + tol, wl_t)])
    psel = findall(t -> t0 < t <= t1 + tol, pred_t)
    pr   = sum(pred_wall[psel])
    s = cfd + pr
    global hybrid_total += s
    println("  hybrid #$k  CFD=$(round(cfd, digits=2)) s + pred=$(round(pr, digits=3)) s = $(round(s, digits=2)) s  ($(round(s/60, digits=3)) min)  [$(length(psel)) preds]")
end
println("  total hybrid sections: $(round(hybrid_total, digits=2)) s  ($(round(hybrid_total/60, digits=3)) min)")

# Per-rollout cycle: the block of ~n_switch (150) WaterLily steps run between rollouts
# vs the rollout's own wall time. Grouped by index in hybrid_time_wat (res.pred_idx are
# the rollout boundaries), so no timestamp tolerance is needed. The 150th iteration of
# each cycle is the rollout itself, so a full block holds 149 CFD steps; the first block
# of a section can be shorter (hs.step resets to 0 after a retrain, firing a rollout
# immediately). n_integrs[i] is how many CFD steps rollout i fast-forwarded over.
println("\n--- CFD block between rollouts  vs  rollout wall time ---")
for (k, (t0, t1)) in enumerate(hybrid_spans)
    ks = findall(t -> t0 < t <= t1 + tol, pred_t)   # rollouts in this section, in order
    sec_lo = findfirst(j -> res.hybrid_time_wat[wl_idx[j]] > t0, eachindex(wl_idx))
    prevbound = wl_idx[sec_lo] - 1                   # last hybrid_time_wat index before section
    println("  hybrid section #$k:")
    for (b, i) in enumerate(ks)
        pe = res.pred_idx[i]
        between = findall(j -> prevbound < wl_idx[j] < pe, eachindex(wl_idx))
        w = sum(wl_wall[between])
        println("    cycle $b: $(rpad(length(between),3)) CFD steps = $(rpad(round(w, digits=3),5)) s  →  rollout $(rpad(round(pred_wall[i]*1000, digits=1),6)) ms  (skipped $(n_integrs[i]) steps)")
        prevbound = pe
    end
end

print_timing_summary(base_dir)

ref_drag = first.(res.forces_ref)
ref_lift = last.(res.forces_ref)

hyb_drag = first.(res.hybrid_forces_wat)
hyb_lift = last.(res.hybrid_forces_wat)


# meanplot = plot_meanflow_comparison(sim_meanflow, ref_meanflow;
#     savedir=joinpath(base_dir, "meanflow_panels"))
# display(meanplot)

# rst_plot = plot_rst_comparison(sim_meanflow, ref_meanflow; style=:filled,
#     savedir=joinpath(base_dir, "rst_panels"))
# display(rst_plot)

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

nothing
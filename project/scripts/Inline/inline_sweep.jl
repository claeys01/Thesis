using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using Printf

# ─────────────────────────────────────────────────────────────────
# One-factor-at-a-time sweep around a baseline.
# Baseline runs once; for each axis, we try the two NON-baseline values.
# Total runs = 1 + sum(length(values) - 1)  (= 1 + 6×2 = 13 for 3 values each).
# t_cutoff_extra = sim time of fresh WaterLily data after retrain flag.
# ─────────────────────────────────────────────────────────────────
baseline = (
    t_run             = 20,
    ae_epochs         = 1000,
    node_iters        = 250,
    ae_retrain_epochs = 300,
    node_retrain_iters = 100,
    t_cutoff_extra    = 10.0,
)

axis_values = (
    t_run              = [15, 20, 25],
    ae_epochs          = [500, 1000, 1500],
    node_iters         = [150, 250, 400],
    ae_retrain_epochs  = [150, 300, 500],
    node_retrain_iters = [50, 100, 200],
    t_cutoff_extra     = [5.0, 10.0, 15.0],
)

# Build the OFAT sweep: baseline first, then per-axis variants.
sweep = NamedTuple[]
push!(sweep, merge((; axis=:baseline), baseline))
for axis in keys(axis_values)
    for v in axis_values[axis]
        v == baseline[axis] && continue   # skip the baseline value on each axis
        push!(sweep, merge((; axis=axis), baseline, NamedTuple{(axis,)}((v,))))
    end
end

root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
sweep_root = joinpath(root_path, "data", "inline_sweeps", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(sweep_root)
u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")

# ─────────────────────────────────────────────────────────────────
# Run a single configuration end-to-end. Mirrors inline_noload.jl.
# ─────────────────────────────────────────────────────────────────
function run_inline(p::InlineParams, t_cutoff_extra::Real, savedir::String, u₀)
    mkpath(savedir)
    simdata_path = joinpath(savedir, "U_inline.jld2")
    sim = circle_shedding_biot(; mem=Array, perturb=false)
    hs = HybridState(sim, nothing, p, savedir, nothing, nothing)

    t = (; ae=0.0, node=0.0, ae_re=0.0, node_re=0.0, wl=0.0)
    wl0 = time()
    simdata = run_warmup!(hs, p.t_run; u₀=u₀, save_path=simdata_path)
    t = merge(t, (wl = t.wl + (time() - wl0),))

    ae_args = LuxArgs(epochs=p.ae_epochs, save_path=savedir, λdiv=100.0, λcurl=10.0,
        train_downsample=500, t_training=p.t_train,
        full_data_path=simdata_path, simdata_ram=simdata)
    t0 = time()
    ae_bundle, AE_path = train_AE(ae_args; return_path=true)
    t = merge(t, (ae = time() - t0,))
    normalizer = load_normalizer(AE_path)
    ae_bundle = cpu_device()(ae_bundle)

    node_args = NodeArgs(save_path=savedir, maxiters=p.node_iters,
        extrapolate=false, use_gpu=false, latent_dim=ae_args.latent_dim)
    t0 = time()
    node, node_path = train_NODE(node_args; ae_bundle=ae_bundle,
        normalizer=normalizer, ae_args=ae_args)
    t = merge(t, (node = time() - t0,))

    hs.aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=false)
    hs.AE_path, hs.node_path = AE_path, node_path
    run_hybrid!(hs)

    if hs.retrain_needed
        GC.gc()
        push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Cutoff"))
        wl0 = time()
        simdata = run_warmup!(hs, sim_time(hs.sim) + t_cutoff_extra;
            simdata=simdata, save_path=simdata_path)
        t = merge(t, (wl = t.wl + (time() - wl0),))

        ae_re_args = LuxArgs(η=2e-4, epochs=p.ae_retrain_epochs,
            λdiv=100.0, λcurl=10.0, t_training=simdata.time[end] * 0.8,
            retrain=true, checkpoint_path=AE_path, save_path=savedir,
            full_data_path=simdata_path, simdata_ram=simdata)
        t0 = time()
        ae_re_bundle, AE_re_path = train_AE(ae_re_args; return_path=true)
        re_normalizer = load_normalizer(AE_re_path)
        t = merge(t, (ae_re = time() - t0,))
        ae_re_bundle = cpu_device()(ae_re_bundle)
        GC.gc()

        node_re_args = NodeArgs(save_path=savedir, extrapolate=false,
            latent_dim=ae_args.latent_dim, η=0.005,
            maxiters=p.node_retrain_iters, group_size=20, continuity_term=600,
            downsample=400, retrain=true, multiple_shooting=true,
            use_gpu=false, node_checkpoint=node_path)
        t0 = time()
        node_re, node_re_path = train_NODE(node_re_args; ae_bundle=ae_re_bundle,
            normalizer=re_normalizer, ae_args=ae_re_args)
        t = merge(t, (node_re = time() - t0,))

        hs.aenode = AENODE(ae_re_bundle, node_re, ae_re_args, node_re_args, re_normalizer; verbose=false)
        hs.AE_path, hs.node_path = AE_re_path, node_re_path
        hs.retrain_needed = false; hs.step = 0
        push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Restarted"))
        run_hybrid!(hs)

        if sim_time(hs.sim) < p.t_accel_end
            wl0 = time()
            simdata = run_warmup!(hs, p.t_accel_end; simdata=simdata, save_path=simdata_path)
            t = merge(t, (wl = t.wl + (time() - wl0),))
        end
    end
    return hs, t
end

# ─────────────────────────────────────────────────────────────────
# Errors of hybrid against reference.
# ─────────────────────────────────────────────────────────────────
relerr(a, b) = sqrt(sum(abs2, a .- b)) / max(sqrt(sum(abs2, b)), eps(eltype(b)))

function field_errors(hs::HybridState)
    m = compute_metrics(hs.res)
    drag_rel = m.rel_err.drag_mean
    lift_rel = m.rel_err.lift_rms

    sim_u, sim_v = hs.sim_meanflow.U[:, :, 1], hs.sim_meanflow.U[:, :, 2]
    ref_u, ref_v = hs.ref_meanflow.U[:, :, 1], hs.ref_meanflow.U[:, :, 2]
    u_rel = relerr(sim_u, ref_u)
    v_rel = relerr(sim_v, ref_v)

    τ     = WaterLily.uu(hs.sim_meanflow)
    τ_ref = WaterLily.uu(hs.ref_meanflow)
    uu_rel = relerr(τ[:, :, 1, 1], τ_ref[:, :, 1, 1])
    vv_rel = relerr(τ[:, :, 2, 2], τ_ref[:, :, 2, 2])
    uv_rel = relerr(τ[:, :, 2, 1], τ_ref[:, :, 2, 1])

    return (; drag_rel, lift_rel, u_rel, v_rel, uu_rel, vv_rel, uv_rel)
end

# ─────────────────────────────────────────────────────────────────
# Sweep loop.
# ─────────────────────────────────────────────────────────────────
records = []
for (i, cfg) in enumerate(sweep)
    @info "Sweep $i / $(length(sweep))" cfg
    p = InlineParams(
        t_run = cfg.t_run,
        t_train = round(cfg.t_run * 0.83, digits=3),
        t_accel_end = 50,
        ae_epochs = cfg.ae_epochs,
        ae_retrain_epochs = cfg.ae_retrain_epochs,
        node_iters = cfg.node_iters,
        node_retrain_iters = cfg.node_retrain_iters,
        n_switch = 150, max_retrain_flags = 3, save_interval = 0.25,
    )
    run_dir = joinpath(sweep_root, @sprintf("run_%02d_%s", i, String(cfg.axis)))
    hs, t = run_inline(p, cfg.t_cutoff_extra, run_dir, u₀)
    e = field_errors(hs)
    push!(records, merge((; run=i), cfg, e,
        (; ae_min=t.ae/60, node_min=t.node/60,
           ae_re_min=t.ae_re/60, node_re_min=t.node_re/60,
           wl_min=t.wl/60)))
    jldsave(joinpath(sweep_root, "results.jld2"); records)
    GC.gc()
end

# ─────────────────────────────────────────────────────────────────
# Print summary table.
# ─────────────────────────────────────────────────────────────────
println("\n" * "="^120)
println("OFAT SWEEP SUMMARY  (relative errors, hybrid vs reference)")
println("="^120)
@printf("%-3s %-19s %-6s %-7s %-7s %-9s %-9s %-7s | %-7s %-7s %-6s %-6s %-7s %-7s %-7s\n",
    "i","axis","t_run","ae_ep","node_it","ae_re_ep","nd_re_it","cut+",
    "drag%","lift%","u","v","uu","vv","uv")
println("-"^120)
for r in records
    @printf("%-3d %-19s %-6.1f %-7d %-7d %-9d %-9d %-7.1f | %-7.2f %-7.2f %-6.3f %-6.3f %-7.3f %-7.3f %-7.3f\n",
        r.run, String(r.axis), r.t_run, r.ae_epochs, r.node_iters, r.ae_retrain_epochs,
        r.node_retrain_iters, r.t_cutoff_extra,
        r.drag_rel, r.lift_rel, r.u_rel, r.v_rel, r.uu_rel, r.vv_rel, r.uv_rel)
end
println("="^120)
@info "Sweep complete" sweep_root n_runs=length(records)

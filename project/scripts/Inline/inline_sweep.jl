using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using Printf

# Four-axis sweep, one-at-a-time around the default point
#   (ae_epochs=500, latent_dim=16, node_iters=250, group_size=20):
#   1) AE training epochs: 100, 250, 500
#   2) Latent dimension:    8,  16,  32
#   3) NODE iters:        100, 250, 500
#   4) Group size:         10,  20,  40
# The default point is shared across all four axes.
fixed = (
    t_run              = 20,
    ae_retrain_epochs  = 100,
    node_retrain_iters = 100,
    t_cutoff_extra     = 10.0,
)

configs = [
    # (sweep = "ae_epochs",  ae_epochs = 100, latent_dim = 16, node_iters = 250, group_size = 20),
    # (sweep = "ae_epochs",  ae_epochs = 250, latent_dim = 16, node_iters = 250, group_size = 20),
    # (sweep = "ae_epochs",  ae_epochs = 500, latent_dim = 16, node_iters = 250, group_size = 20),  # shared default point
    # (sweep = "latent_dim", ae_epochs = 500, latent_dim = 8,  node_iters = 250, group_size = 20),
    # (sweep = "latent_dim", ae_epochs = 500, latent_dim = 32, node_iters = 250, group_size = 20),
    (sweep = "node_iters", ae_epochs = 500, latent_dim = 16, node_iters = 100, group_size = 20),
    (sweep = "node_iters", ae_epochs = 500, latent_dim = 16, node_iters = 500, group_size = 20),
    (sweep = "group_size", ae_epochs = 500, latent_dim = 16, node_iters = 250, group_size = 10),
    (sweep = "group_size", ae_epochs = 500, latent_dim = 16, node_iters = 250, group_size = 40),
]

root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
sweep_root = joinpath(root_path, "data", "inline_sweeps", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(sweep_root)
u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")

function run_inline(p::InlineParams, latent_dim::Int, t_cutoff_extra::Real, savedir::String, u₀)
    mkpath(savedir)
    simdata_path = joinpath(savedir, "U_inline.jld2")
    sim = circle_shedding_biot(; mem=Array, perturb=false)
    hs = HybridState(sim, nothing, p, savedir, nothing, nothing)

    t = (; ae=0.0, node=0.0, ae_re=0.0, node_re=0.0, wl=0.0)
    wl0 = time()
    simdata = run_warmup!(hs, p.t_run; u₀=u₀, save_path=simdata_path)
    t = merge(t, (wl = t.wl + (time() - wl0),))

    ae_args = LuxArgs(epochs=p.ae_epochs, latent_dim=latent_dim, save_path=savedir,n_dense=1,
        λdiv=100.0, λcurl=100.0, train_downsample=500, t_training=p.t_train,
        full_data_path=simdata_path, simdata_ram=simdata)
    t0 = time()
    ae_bundle, AE_path = train_AE(ae_args; return_path=true)
    t = merge(t, (ae = time() - t0,))
    normalizer = load_normalizer(AE_path)
    ae_bundle = cpu_device()(ae_bundle)

    node_args = NodeArgs(save_path=savedir, maxiters=p.node_iters,
        group_size=p.group_size,
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
            λdiv=100.0, λcurl=100.0, t_training=simdata.time[end] * 0.8, n_dense=1,
            retrain=true, checkpoint_path=AE_path, save_path=savedir, train_downsample=500,
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
    save_results(hs)
    return hs, t
end

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

records = []
for (i, cfg) in enumerate(configs)
    @info "Sweep $i / $(length(configs))" cfg
    p = InlineParams(
        t_run = fixed.t_run,
        t_train = round(fixed.t_run * 0.83, digits=3),
        t_accel_end = 50,
        ae_epochs = cfg.ae_epochs,
        ae_retrain_epochs = fixed.ae_retrain_epochs,
        node_iters = cfg.node_iters,
        node_retrain_iters = fixed.node_retrain_iters,
        group_size = cfg.group_size,
        n_switch = 150, max_retrain_flags = 3, save_interval = 0.25,
    )
    run_name = @sprintf("ae%d_lat%d_nit%d_gs%d",
        cfg.ae_epochs, cfg.latent_dim, cfg.node_iters, cfg.group_size)
    run_dir  = joinpath(sweep_root, run_name)
    hs, t = run_inline(p, cfg.latent_dim, fixed.t_cutoff_extra, run_dir, u₀)
    e = field_errors(hs)
    push!(records, merge((; run=i, name=run_name), cfg, e,
        (; ae_min=t.ae/60, node_min=t.node/60,
           ae_re_min=t.ae_re/60, node_re_min=t.node_re/60,
           wl_min=t.wl/60)))
    jldsave(joinpath(sweep_root, "results.jld2"); records)
    GC.gc()
end

println("\n" * "="^130)
println("INLINE SWEEP SUMMARY  (relative errors, hybrid vs reference)")
println("="^130)
@printf("%-3s %-12s %-6s %-6s %-6s %-6s | %-7s %-7s %-6s %-6s %-7s %-7s %-7s\n",
    "i", "sweep", "ae_ep", "latent", "nit", "gs",
    "drag%", "lift%", "u", "v", "uu", "vv", "uv")
println("-"^130)
for r in records
    @printf("%-3d %-12s %-6d %-6d %-6d %-6d | %-7.2f %-7.2f %-6.3f %-6.3f %-7.3f %-7.3f %-7.3f\n",
        r.run, r.sweep, r.ae_epochs, r.latent_dim, r.node_iters, r.group_size,
        r.drag_rel, r.lift_rel, r.u_rel, r.v_rel, r.uu_rel, r.vv_rel, r.uv_rel)
end
println("="^130)
@info "Sweep complete" sweep_root n_runs=length(records)

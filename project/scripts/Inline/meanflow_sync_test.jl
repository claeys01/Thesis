#
# Mean-flow synchronisation test.
#
# Goal: prove (empirically) whether the hybrid `sim_meanflow` and the reference
# `ref_meanflow` are updated with the SAME number of snapshots, each absorbed at
# the SAME simulation-time instance.
#
# Both `MeanFlow` objects keep a running `t` vector: every call that folds a new
# snapshot into the average pushes that snapshot's time onto `meanflow.t`
# (see WaterLily.update! and Thesis.update_meanflow_snapshot!). That vector is
# therefore the ground-truth record of *when* and *how often* each mean flow was
# updated. After running a short-but-representative hybrid simulation we simply
# line `sim_meanflow.t` up against `ref_meanflow.t` and report the comparison.
#
# Run from the repo root:
#   julia --project=project project/scripts/Inline/meanflow_sync_test.jl
#

using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Printf

root_path = is_hpc() ? "/scratch/mfbclaeys" : ""

# Pretrained AE used by inline_loadAE.jl — only used to build a working AENODE so
# the hybrid prediction branch is exercised; its quality is irrelevant to the test.
AE_path = "data/saved_models/inline_runs_hpc/latent_epoch_sweep/ae_epochs_100_latent_16/AE_May24-0724__E100_HW256x256_C4to2_nc6_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE_path = joinpath(root_path, AE_path)

# Short window: warmup builds enough data to fit a throwaway NODE, then the hybrid
# phase runs long enough to fire several predictions and cross many save points.
params = InlineParams(
    t_run        = 6.0,    # warmup end (tU/L) — also the AE/NODE training window
    t_accel_end  = 9.0,    # hybrid end (tU/L)
    node_iters   = 25,     # throwaway NODE, just needs to produce predictions
    n_switch     = 30,     # attempt a prediction every n_switch hybrid steps
    pred_Δt      = 0.35,
    save_interval = 0.1,   # mean-flow update cadence
    sample_interval = 0.0, # collect a training snapshot every CFD step
    max_retrain_flags = typemax(Int),  # never bail out into the retrain branch
)

savedir = joinpath(root_path, "data", "inline_runs",
    "meanflow_sync_test_" * Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline.jld2")

u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
sim = circle_shedding_biot(; mem=Array, perturb=false)

hs = HybridState(sim, nothing, params, savedir, AE_path, nothing)

@info "── Warmup (builds reference + hybrid mean flows in lock-step) ──"
simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path, verbose=false)

@info "── Building a throwaway AENODE so the hybrid prediction branch runs ──"
normalizer = load_normalizer(AE_path)
ae_bundle, ae_args = load_trained_AE(AE_path)
ae_args.train_downsample = 300
ae_args.full_data_path = simdata_path

node_args = NodeArgs(
    save_path = savedir,
    maxiters = params.node_iters,
    downsample = ae_args.train_downsample,
    group_size = 15,
    continuity_term = 250,
    extrapolate = false,
    use_gpu = false,
    latent_dim = ae_args.latent_dim,
)
node, node_path = train_NODE(node_args; ae_bundle=ae_bundle, normalizer=normalizer, ae_args=ae_args)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=false)
hs.aenode = aenode
hs.node_path = node_path

@info "── Hybrid phase (predictions + mean-flow updates) ──"
run_hybrid!(hs)

# ----------------------------------------------------------------------------
# Verification: compare the two mean-flow update timelines.
# ----------------------------------------------------------------------------
function verify_meanflow_sync(hs; tol = 1f-4)
    th  = hs.sim_meanflow.t   # hybrid mean-flow update times
    tr  = hs.ref_meanflow.t   # reference mean-flow update times

    nh, nr = length(th), length(tr)
    # t[1] is the shared t_init set at construction; updates = length - 1.
    uh, ur = nh - 1, nr - 1
    n = min(nh, nr)

    diffs = Float64.(abs.(th[1:n] .- tr[1:n]))
    max_diff = isempty(diffs) ? 0.0 : maximum(diffs)
    n_mismatch = count(>(tol), diffs)
    first_div = findfirst(>(tol), diffs)

    monotonic_h = all(diff(th) .> 0)
    monotonic_r = all(diff(tr) .> 0)

    println("\n" * "="^74)
    println("MEAN-FLOW SYNCHRONISATION REPORT")
    println("="^74)

    println("\nContext")
    println("-"^74)
    @printf("  Hybrid phase predictions fired : %d\n", length(hs.n_integrs))
    if !isempty(hs.n_integrs)
        @printf("  Avg NODE steps per prediction  : %.1f\n", mean(hs.n_integrs))
    end
    @printf("  save_interval (tU/L)           : %s\n", string(hs.params.save_interval))
    @printf("  sim_time at end (tU/L)         : %.4f\n", sim_time(hs.sim))
    @printf("  MeanFlow.t units               : WaterLily flow-time  (= sim_time * L/U)\n")
    @printf("  MeanFlow.t range (hybrid)      : [%.4f, %.4f]\n", first(th), last(th))
    @printf("  MeanFlow.t range (reference)   : [%.4f, %.4f]\n", first(tr), last(tr))

    println("\nUpdate counts (number of snapshots folded into each mean flow)")
    println("-"^74)
    @printf("  Hybrid    sim_meanflow : %d updates  (%d t-entries incl. t_init)\n", uh, nh)
    @printf("  Reference ref_meanflow : %d updates  (%d t-entries incl. t_init)\n", ur, nr)
    @printf("  Same number of snapshots? %s\n", uh == ur ? "YES ✓" : "NO ✗  (Δ = $(abs(uh - ur)))")

    println("\nSide-by-side update times  (idx | hybrid t | reference t | |Δ| | match)")
    println("-"^74)
    for i in 1:n
        ok = diffs[i] <= tol
        @printf("  %4d | %12.5f | %12.5f | %10.3e | %s\n",
            i, th[i], tr[i], diffs[i], ok ? "ok" : "MISMATCH")
    end
    if nh > n
        println("  -- hybrid has $(nh - n) extra entry(ies) with no reference counterpart --")
        for i in (n+1):nh
            @printf("  %4d | %12.5f | %12s | %10s | EXTRA(hybrid)\n", i, th[i], "—", "—")
        end
    elseif nr > n
        println("  -- reference has $(nr - n) extra entry(ies) with no hybrid counterpart --")
        for i in (n+1):nr
            @printf("  %4d | %12s | %12.5f | %10s | EXTRA(ref)\n", i, "—", tr[i], "—")
        end
    end

    println("\nSummary")
    println("-"^74)
    @printf("  Aligned entries compared       : %d\n", n)
    @printf("  Time tolerance                 : %.1e\n", tol)
    @printf("  Max |Δt| over aligned entries  : %.3e\n", max_diff)
    @printf("  Mismatched time instances      : %d\n", n_mismatch)
    if !isnothing(first_div)
        @printf("  First divergence at index      : %d (hybrid=%.5f, ref=%.5f)\n",
            first_div, th[first_div], tr[first_div])
    end
    @printf("  sim_meanflow.t strictly increasing : %s\n", monotonic_h ? "yes" : "NO")
    @printf("  ref_meanflow.t strictly increasing : %s\n", monotonic_r ? "yes" : "NO")

    counts_ok = (uh == ur)
    times_ok  = (n_mismatch == 0) && (nh == nr)
    verdict = counts_ok && times_ok

    println("\n" * "="^74)
    if verdict
        println("VERDICT: PASS ✓  — both mean flows updated with the same number of")
        println("         snapshots, each at the same simulation-time instance.")
    else
        println("VERDICT: FAIL ✗  — the two mean flows are NOT in sync.")
        !counts_ok && println("         · update counts differ ($uh hybrid vs $ur reference)")
        !times_ok  && println("         · $n_mismatch update time(s) do not match within tolerance")
    end
    # `update_meanflow_snapshot!` would push a small dimensionless sim_time onto a
    # vector otherwise growing in (much larger) flow-time units, breaking
    # monotonicity. A strictly increasing, fully-matched timeline therefore means
    # the NODE-predicted snapshots were never folded in.
    if monotonic_h && verdict && !isempty(hs.n_integrs)
        println("\n  NOTE: sim_meanflow.t is strictly increasing and identical to the")
        println("  reference, so NONE of the NODE-predicted snapshots were actually")
        println("  folded into the hybrid mean flow (they hit dt<=0 and were skipped).")
        println("  See update_meanflow_snapshot! / units of t_meanflow vs MeanFlow.t.")
    end
    println("="^74 * "\n")

    return (; verdict, counts_ok, times_ok, uh, ur, max_diff, n_mismatch,
            th = collect(th), tr = collect(tr))
end

result = verify_meanflow_sync(hs)

report_path = joinpath(savedir, "meanflow_sync.jld2")
let th = result.th, tr = result.tr, verdict = result.verdict
    @save report_path th tr verdict
end
@info "Saved mean-flow timelines" report_path verdict=result.verdict

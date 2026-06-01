using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using Printf


root_path = ""
if is_hpc()
    root_path = "/scratch/mfbclaeys"
    # Log job info
    @info "Starting HPC AE+NODE retrain pipeline"
    @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
    @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
    @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
    @info "  Hostname: $(gethostname())"
    @info "  Julia threads: $(Threads.nthreads())"
end

params = InlineParams()

savedir = joinpath(root_path, "data", "inline_runs","RE250", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline_RE250.jld2")

u₀ = load_u0(joinpath(root_path, "data/initial_fields/RE250/2e8/u_0.jld2"))
sim = circle_shedding_biot(; Re=250, mem=Array, perturb=false)

hs = HybridState(sim, nothing, params, savedir, nothing, nothing)

wl_warmup_start = time()
simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path)
wl_warmup_elapsed = round((time() - wl_warmup_start) / 60; digits=2)
@info "WaterLily warmup complete" elapsed_min=wl_warmup_elapsed t_simulated=params.t_run

# ================================ Step 1: Train Autoencoder ================================
@info "── Step 1/4: Training Autoencoder ──"
ae_start = time()

ae_args = LuxArgs(
        λdiv=Float64(0.0),
        λcurl=Float64(0.0),
        epochs=hs.params.ae_epochs, 
        save_path=savedir,
        train_downsample=hs.params.downsample,
        t_training=hs.params.t_train,
        full_data_path=simdata_path, 
        simdata_ram=simdata,
    )

ae_bundle, AE_path = train_AE(ae_args; return_path=true)
ae_elapsed = round((time() - ae_start) / 60; digits=1)
@info "AE initial training complete" elapsed_min=ae_elapsed checkpoint=AE_path

# ae_args.simdata_ram = nothing   # release the simdata ref
normalizer = load_normalizer(AE_path)

ae_bundle = cpu_device()(ae_bundle)

# ================================ Step 2: Train NODE ================================
@info "── Step 2/4: Training Neural ODE ──"
node_args = NodeArgs(
        save_path=savedir,
        maxiters = params.node_iters,
        downsample=ae_args.train_downsample,
        group_size=hs.params.group_size,
        continuity_term=hs.params.continuity_term,
        latent_dim = ae_args.latent_dim,
    )
node_start = time()
node, node_path = train_NODE(
    node_args;
    ae_bundle = ae_bundle,
    normalizer = normalizer,
    ae_args = ae_args,
)
node_elapsed = round((time() - node_start) / 60; digits=1)
@info "NODE training complete" elapsed_min=node_elapsed node_path=node_path

# ae_bundle = cpu_device()(ae_bundle)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

# hs = HybridState(sim, aenode, params, savedir, AE_path_tl1, node_path)
hs.aenode = aenode
hs.AE_path = AE_path
hs.node_path = node_path

# Retrain/cutoff timing accumulators. Declared outside the loop so they survive
# across iterations and are visible in the timing summary below.
ae_retrain_elapsed_total = 0.0
node_retrain_elapsed_total = 0.0
wl_cutoff_elapsed_total = 0.0
retrain_timings = NamedTuple[]   # one entry per completed retrain cycle

# run_hybrid!(hs)
while sim_time(hs.sim) < hs.params.t_accel_end
    run_hybrid!(hs)
    global simdata, ae_retrain_elapsed_total, node_retrain_elapsed_total, wl_cutoff_elapsed_total

    sim_time(hs.sim) >  hs.params.t_accel_end && break
    if hs.retrain_needed
        GC.gc()
        @info "Retraining triggered at sim_time=$(sim_time(hs.sim)), step=$(hs.step)"
        push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Cutoff"))

        println("continueing to run simulation without AENODE")

        wl_cutoff_start = time()
        t_before = sim_time(hs.sim)
        simdata = run_warmup!(hs, sim_time(hs.sim) + hs.params.t_update; simdata=simdata, save_path=simdata_path)

        wl_cutoff_elapsed = round((time() - wl_cutoff_start) / 60; digits=2)
        wl_cutoff_elapsed_total += wl_cutoff_elapsed
        @info "WaterLily cutoff run complete" elapsed_min=wl_cutoff_elapsed t_simulated=(sim_time(hs.sim) - t_before)

        sim_time(hs.sim) >  hs.params.t_accel_end && break

        # ================================ Step 3: Retrain AE ================================
        ae_retrain_start = time()
        ae_retrain_args = LuxArgs(
            η = 2e-4,
            λdiv=Float64(0.0),
            λcurl=Float64(0.0),
            epochs=hs.params.ae_retrain_epochs, 
            t_training=simdata.time[end] * 0.85 ,
            train_downsample=hs.params.downsample,
            retrain=true,
            checkpoint_path=hs.AE_path,
            save_path=savedir,
            full_data_path=simdata_path, 
            simdata_ram=simdata,
        )
        
        ae_retrain_bundle, AE_retrain_path = train_AE(ae_retrain_args; return_path=true)
        retrain_normalizer = load_normalizer(AE_retrain_path)
        ae_retrain_elapsed = round((time() - ae_retrain_start) / 60; digits=1)
        ae_retrain_elapsed_total += ae_retrain_elapsed
        @info "AE retraining complete" elapsed_min=ae_retrain_elapsed checkpoint=AE_path

        # ================================ Step 4: Retrain NODE ================================

        @info "── Step 4/4: Retraining Neural ODE ──"
        ae_retrain_bundle = cpu_device()(ae_retrain_bundle)
        GC.gc()
        node_retrain_start = time()
        node_retrain_args = NodeArgs(
            save_path=savedir,
            latent_dim = ae_args.latent_dim,
            η = 0.01,              # lower LR for fine-tuning
            maxiters = hs.params.node_retrain_iters,          # more iterations
            group_size = hs.params.group_size,         # keep tighter segments
            continuity_term = hs.params.continuity_term_retrain,   # stronger continuity for stability
            downsample = hs.params.downsample,  
            retrain = true,
            multiple_shooting = true,
            node_checkpoint = node_path,
        )
            
        node_retrain, node_retrain_path = train_NODE(node_retrain_args;
            ae_bundle = ae_retrain_bundle,
            normalizer = retrain_normalizer, ae_args = ae_retrain_args,
        )
        node_retrain_elapsed = round((time() - node_retrain_start) / 60; digits=1)
        node_retrain_elapsed_total += node_retrain_elapsed
        @info "NODE retraining complete" elapsed_min=node_retrain_elapsed node_path=node_retrain_path

        push!(retrain_timings, (wl_cutoff=wl_cutoff_elapsed, ae=ae_retrain_elapsed, node=node_retrain_elapsed))

        hs.aenode = AENODE(ae_retrain_bundle, node_retrain, ae_retrain_args, node_retrain_args, retrain_normalizer; verbose=true)
        hs.AE_path = AE_retrain_path
        hs.node_path = node_retrain_path
        hs.retrain_needed = false
        hs.step = 0

        push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Restarted"))
    end
end

save_results(hs)

# ================================ Timing Summary ================================
# Retrain/cutoff totals are accumulated inside the loop above. wl_tail is
# optional and may never be set, so keep its guard.
wl_tail_elapsed_total = @isdefined(wl_tail_elapsed) ? wl_tail_elapsed : 0.0
n_retrains = length(retrain_timings)

ml_total = ae_elapsed + node_elapsed + ae_retrain_elapsed_total + node_retrain_elapsed_total
wl_total = wl_warmup_elapsed + wl_cutoff_elapsed_total + wl_tail_elapsed_total
grand_total = ml_total + wl_total

println("\n" * "="^60)
println("              TRAINING vs WATERLILY TIMING SUMMARY")
println("="^60)
@printf("ML training\n")
@printf("  AE initial train     : %7.2f min\n", ae_elapsed)
@printf("  NODE initial train   : %7.2f min\n", node_elapsed)
@printf("  Retrain iterations   : %7d\n", n_retrains)
@printf("  AE retrain   (total) : %7.2f min\n", ae_retrain_elapsed_total)
@printf("  NODE retrain (total) : %7.2f min\n", node_retrain_elapsed_total)
@printf("  ML subtotal          : %7.2f min\n", ml_total)
println("-"^60)
@printf("WaterLily simulation\n")
@printf("  Warmup run           : %7.2f min\n", wl_warmup_elapsed)
@printf("  Cutoff run   (total) : %7.2f min\n", wl_cutoff_elapsed_total)
@printf("  Tail run             : %7.2f min\n", wl_tail_elapsed_total)
@printf("  WaterLily subtotal   : %7.2f min\n", wl_total)
println("-"^60)
if n_retrains > 0
    @printf("Per-retrain breakdown\n")
    for (i, t) in enumerate(retrain_timings)
        @printf("  [%2d] cutoff %6.2f | AE %6.2f | NODE %6.2f  (min)\n",
                i, t.wl_cutoff, t.ae, t.node)
    end
    println("-"^60)
end
@printf("Grand total            : %7.2f min\n", grand_total)
if wl_total > 0
    @printf("ML / WaterLily ratio   : %7.2fx\n", ml_total / wl_total)
end
println("="^60)

jldsave(joinpath(savedir, "timing_summary.jld2");
    ae_elapsed_min          = ae_elapsed,
    node_elapsed_min        = node_elapsed,
    n_retrains              = n_retrains,
    ae_retrain_elapsed_min  = ae_retrain_elapsed_total,
    node_retrain_elapsed_min = node_retrain_elapsed_total,
    wl_warmup_elapsed_min   = wl_warmup_elapsed,
    wl_cutoff_elapsed_min   = wl_cutoff_elapsed_total,
    wl_tail_elapsed_min     = wl_tail_elapsed_total,
    ml_total_min            = ml_total,
    wl_total_min            = wl_total,
    grand_total_min         = grand_total,
    retrain_timings         = retrain_timings,
)
@info "Timing summary saved" path=joinpath(savedir, "timing_summary.jld2")

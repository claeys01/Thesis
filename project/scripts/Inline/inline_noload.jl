using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots


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

params = params = InlineParams(
        t_run = 20, 
        t_train = 16.603,
        t_accel_end = 50,
        ae_epochs = 1000,
        ae_retrain_epochs = 300,
        node_iters = 250,
        node_retrain_iters = 100,
        n_switch = 150,
        max_retrain_flags = 3,
        save_interval = 0.25, # needs to be fixed still, 
    )


savedir = joinpath(root_path, "data", "inline_runs", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline.jld2")

u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
sim = circle_shedding_biot(; mem=Array, perturb=false)

hs = HybridState(sim, nothing, params, savedir, nothing, nothing)

simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path)

# ================================ Step 1: Train Autoencoder ================================
@info "── Step 1/4: Training Autoencoder ──"
ae_start = time()

div = 100.0
curl = 10.0
@info "AE hyperparameters" epochs=params.ae_epochs λdiv=div λcurl=curl

ae_args = LuxArgs(
        epochs=params.ae_epochs, 
        save_path=savedir,
        λdiv=Float64(div), 
        λcurl=Float64(curl),
        train_downsample=500,
        t_training=params.t_train,
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
        extrapolate = false,
        use_gpu = false,
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

# @info "Steps 1-2 complete" elapsed_min=round((time() - total_start) / 60; digits=1)
# ae_bundle = cpu_device()(ae_bundle)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

# hs = HybridState(sim, aenode, params, savedir, AE_path_tl1, node_path)
hs.aenode = aenode
hs.AE_path = AE_path
hs.node_path = node_path

run_hybrid!(hs)

if hs.retrain_needed
    GC.gc()
    @info "Retraining triggered at sim_time=$(sim_time(hs.sim)), step=$(hs.step)"
    push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Cutoff"))

    println("continueing to run simulation without AENODE")

    simdata = run_warmup!(hs, sim_time(hs.sim) + 10; simdata=simdata, save_path=simdata_path)

    # ================================ Step 3: Retrain AE ================================
    ae_retrain_start = time()
    ae_retrain_args = LuxArgs(
        η = 2e-4,
        epochs=params.ae_retrain_epochs, 
        λdiv=Float64(div), 
        λcurl=Float64(curl),
        t_training=simdata.time[end] * 0.8 ,
        retrain=true,
        checkpoint_path=AE_path,
        save_path=savedir,
        full_data_path=simdata_path, 
        simdata_ram=simdata,
    )
    
    ae_retrain_bundle, AE_retrain_path = train_AE(ae_retrain_args; return_path=true)
    retrain_normalizer = load_normalizer(AE_retrain_path)
    ae_retrain_elapsed = round((time() - ae_retrain_start) / 60; digits=1)
    @info "AE retraining complete" elapsed_min=ae_retrain_elapsed checkpoint=AE_path

    # ================================ Step 4: Retrain NODE ================================

    @info "── Step 4/4: Retraining Neural ODE ──"
    ae_retrain_bundle = cpu_device()(ae_retrain_bundle)
    GC.gc()
    node_retrain_start = time()
    node_retrain_args = NodeArgs(
            save_path=savedir,
            extrapolate = false,
            latent_dim = ae_args.latent_dim,
            η = 0.01,              # lower LR for fine-tuning
            maxiters = params.node_retrain_iters,          # more iterations
            group_size = 20,         # keep tighter segments
            continuity_term = 500,   # stronger continuity for stability
            downsample = 400,  
            retrain = true,
            multiple_shooting = true,
            use_gpu = false, 
            node_checkpoint = node_path,
        )
        
    node_retrain, node_retrain_path = train_NODE(node_retrain_args;
        ae_bundle = ae_retrain_bundle,
        normalizer = retrain_normalizer, ae_args = ae_retrain_args,
    )
    node_retrain_elapsed = round((time() - node_retrain_start) / 60; digits=1)
    @info "NODE retraining complete" elapsed_min=node_retrain_elapsed node_path=node_retrain_path


    hs.aenode = AENODE(ae_retrain_bundle, node_retrain, ae_retrain_args, node_retrain_args, retrain_normalizer; verbose=true)
    hs.AE_path = AE_retrain_path
    hs.node_path = node_retrain_path
    hs.retrain_needed = false
    hs.step = 0
    
    push!(hs.mode_log, (t_start=sim_time(hs.sim), t_end=sim_time(hs.sim), mode="Restarted"))
    run_hybrid!(hs)

    if sim_time(hs.sim) < params.t_accel_end
        simdata = run_warmup!(hs, params.t_accel_end; simdata=simdata, save_path=simdata_path)
    end
end

save_results(hs)

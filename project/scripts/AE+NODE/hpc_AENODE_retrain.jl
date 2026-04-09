#!/usr/bin/env julia

# Set HPC-specific environment variables BEFORE loading packages
# ENV["THESIS_HPC"] = "true"         # Mark as HPC environment
# ENV["THESIS_USE_CUDA"] = "true"  # Uncomment if using GPU nodes

# Activate the project
# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))

# Load the module
using Thesis
using JLD2

function main()
    total_start = time()
    
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

   
    # ================================ Step 1: Train Autoencoder ================================
    @info "── Step 1/4: Training Autoencoder ──"
    ae_start = time()

    tl_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_full.jld2")    
    @info "Loading training data from: $tl_path"

    div = 1000.0
    curl = 100.0
    epochs = 400
    @info "AE hyperparameters" epochs=epochs λdiv=div λcurl=curl

    ae_args = LuxArgs(
            epochs=epochs, 
            λdiv=Float64(div), 
            λcurl=Float64(curl), 
            full_data_path=tl_path
        )

    ae, ae_ps, ae_st, AE_path = train_AE(ae_args; return_path=true)
    ae_elapsed = round((time() - ae_start) / 60; digits=1)
    @info "AE initial training complete" elapsed_min=ae_elapsed checkpoint=AE_path

    normalizer = load_normalizer(AE_path)

    # ================================ Step 2: Train NODE ================================
    @info "── Step 2/4: Training Neural ODE ──"
    node_start = time()
    node_path = train_NODE(
        NodeArgs(
            extrapolate = false,
            use_gpu = false,
            latent_dim = ae_args.latent_dim,
        );
        ae = ae,
        ae_ps = ae_ps,
        ae_st = ae_st,
        normalizer = normalizer,
        ae_args = ae_args,
    )
    node_elapsed = round((time() - node_start) / 60; digits=1)
    @info "NODE training complete" elapsed_min=node_elapsed node_path=node_path

    @info "Steps 1-2 complete" elapsed_min=round((time() - total_start) / 60; digits=1)
    GC.gc()

    # ================================ Simulation / AENODE integration ================================
    # ─── (placeholder: simulation running & AENODE integration would go here) ───
    # ─── retrain criteria is triggered ───

    
    # ================================ Step 3: Retrain AE ================================
    @info "Transfer learning criterion triggered"
    @info "── Step 3/4: Retraining AE ──"
    ae_retrain_start = time()

    checkpoint = JLD2.load(AE_path)
    args_dict = checkpoint["args"]
    # Ensure keys are Symbols for keyword splatting
    # if eltype(keys(args_dict)) <: AbstractString
        # args_dict = Dict(Symbol(k) => v for (k, v) in args_dict)
    # end
    ae_args = LuxArgs(; args_dict...)

    retraindata = load_simdata(tl_path)
    @info "Retrain data loaded" n_snapshots=length(retraindata.time) t_end=retraindata.time[end]

    ae_args.η = 2e-4
    ae_args.epochs = 200
    ae_args.retrain = true
    ae_args.checkpoint_path = AE_path
    ae_args.full_data_path = tl_path
    ae_args.t_training = retraindata.time[end] * 0.8
    ae_args.test_loss = true
    @info "AE retrain hyperparameters" η=ae_args.η epochs=ae_args.epochs t_training=ae_args.t_training

    ae, ae_ps, ae_st, AE_path = train_AE(ae_args; return_path=true)
    ae_retrain_elapsed = round((time() - ae_retrain_start) / 60; digits=1)
    @info "AE retraining complete" elapsed_min=ae_retrain_elapsed checkpoint=AE_path

    # ================================ Step 4: Retrain NODE ================================
    @info "── Step 4/4: Retraining Neural ODE ──"
    GC.gc()
    node_retrain_start = time()

    normalizer = load_normalizer(AE_path)

    node_retrain_path = train_NODE(
        NodeArgs(
            extrapolate = false,
            use_gpu = false,
            latent_dim = ae_args.latent_dim,
            η = 0.001,
            maxiters = 100,
            retrain = true,
            node_checkpoint = node_path,
        );
        ae = ae,
        ae_ps = ae_ps,
        ae_st = ae_st,
        normalizer = normalizer,
        ae_args = ae_args,
    )
    node_retrain_elapsed = round((time() - node_retrain_start) / 60; digits=1)
    @info "NODE retraining complete" elapsed_min=node_retrain_elapsed node_path=node_retrain_path

    @info "Pipeline complete" total_min=round((time() - total_start) / 60; digits=1)
   
end

main()

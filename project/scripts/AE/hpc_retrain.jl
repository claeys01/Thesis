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
    
    root_path = ""
    if is_hpc()
        root_path = "/scratch/mfbclaeys"
        # Log job info
        @info "Starting HPC AE training job"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
        @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
        @info "  Hostname: $(gethostname())"   
    end

    # tl_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_transfer.jld2")
    tl_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_full.jld2")    
    @info "Loading training data from: $tl_path"

    # node_path = joinpath(root_path, "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2")div = 1000.0
    div = 1000.0
    curl = 100.0
    epochs = 1000
    println("\nTraining AE for $epochs epochs with λdiv=$(div), λcurl=$(curl))")

    _, AE_path = train_AE(
        LuxArgs(
            epochs=500, 
            λdiv=Float64(div), 
            λcurl=Float64(curl), 
            full_data_path=tl_path
        ); return_path=true
    )


    @info "Transfer learning criterion triggered"
    @info "Loading checkpoint from: $AE_path"
    
    # Verify files exist
    if !isfile(AE_path)
        @error "Checkpoint not found: $AE_path"
        return
    end
    if !isfile(tl_path)
        @error "Training data not found: $tl_path"
        return
    end
    # aenode = AENODE(AE_path, node_path)
    checkpoint = JLD2.load(AE_path)
    args_dict = checkpoint["args"]
    ae_args = LuxArgs(; args_dict...)

    # @show aenode.ae_args.λdiv, aenode.ae_args.λcurl, aenode.ae_args.λstrain
    # ---- simulation running & AENODE using to integrate
    # ---- retrain criteria is triggered

    retraindata = simdata = load_simdata(tl_path)
    retrain_crit = true
    # test = LuxArgs(aenode.ae_args)
    if retrain_crit
        ae_args.η = 2e-4
        ae_args.epochs = 300
        ae_args.retrain = true
        ae_args.checkpoint_path = AE_path
        ae_args.full_data_path = tl_path
        ae_args.t_training = retraindata.time[end] * 0.8
        ae_args.test_loss = true
        train_AE(ae_args)
    end
end

main()

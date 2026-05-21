#!/usr/bin/env julia

# Set HPC-specific environment variables BEFORE loading packages
ENV["THESIS_HPC"] = "true"         # Mark as HPC environment
# ENV["THESIS_USE_CUDA"] = "true"  # Uncomment if using GPU nodes

# Activate the project
# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))

# Load the module
using Thesis

function main()
    # Log job info
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
    div = 100.0
    curl = 100.0

    epochs = [10, 50, 100, 250, 500, 750, 100]

    savedir = joinpath(root_path, "data", "Lux_models", "epoch_tune")
    for epoch in epochs
        AE_path = train_AE(
            LuxArgs(
                epochs=epoch, 
                λdiv=Float64(div), 
                λcurl=Float64(curl), 
                train_downsample = 500,
                test_loss=true,
                t_training = 25,
                full_data_path=tl_path,
                n_dense=1,
            ); return_path=true
        )
    end
end

main()
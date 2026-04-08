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
        @info "Starting HPC NODE training job"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
        @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
        @info "  Hostname: $(gethostname())"   
    end


    train_latent_path = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_train.jld2")
    test_latent_path  = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_test.jld2")
    total_latent_path = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2")

    train_NODE(
        NodeArgs(
            train_latent_path = train_latent_path,
            test_latent_path = test_latent_path,
            total_latent_path = total_latent_path,
            extrapolate = false,
            use_gpu = true
        )
        )
end

main()

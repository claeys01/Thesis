#!/usr/bin/env julia

# Set HPC-specific environment variables BEFORE loading packages
ENV["THESIS_HPC"] = "true"         # Mark as HPC environment
ENV["THESIS_USE_CUDA"] = "true"  # Uncomment if using GPU nodes

# Activate the project
# using Pkg
# Pkg.activate(joinpath(@__DIR__, ".."))

# Load the module
using Thesis

function main()
    # Log job info
    @info "Starting HPC AE training job"
    @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
    @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
    @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
    @info "  Hostname: $(gethostname())"
    
    # Run training with HPC-appropriate settings


    for div in 1:1000:10000
        for curl in 1:100:1000
            println("\nTraining AE for 200 epochs with λdiv=$(div), λcurl=$(curl))")
            train_AE(LuxArgs(epochs=200, λdiv=Float64(div), λcurl=Float64(curl)))
            println("\n")
        end
    end
end

main()
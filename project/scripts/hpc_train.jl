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
    train_AE(LuxArgs(epochs=200, λdiv=0.0, λcurl=0.0))
    train_AE(LuxArgs(epochs=200, λdiv=1000.0, λcurl=100.0))
    
    train_AE(LuxArgs(epochs=1000, λdiv=0.0, λcurl=0.0))
    train_AE(LuxArgs(epochs=1000,λdiv=1000.0, λcurl=100.0))
end

main()

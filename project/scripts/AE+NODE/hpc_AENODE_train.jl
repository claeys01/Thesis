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
    epochs = 200
    println("\nTraining AE for $epochs epochs with λdiv=$(div), λcurl=$(curl))")


    ae_args = LuxArgs(
            epochs=epochs, 
            λdiv=Float64(div), 
            λcurl=Float64(curl), 
            full_data_path=tl_path
        )
    
    # ── Step 1: Train Autoencoder ──
    @info "── Step 1/2: Training Autoencoder ──"
    ae_start = time()
    ae, ae_ps, ae_st, AE_path = train_AE(ae_args; return_path=true)
    @info "AE training complete" elapsed_min=round((time()-ae_start)/60; digits=1) checkpoint=AE_path

    normalizer = load_normalizer(AE_path)

    # ── Step 2: Train NODE ──
    @info "── Step 2/2: Training Neural ODE ──"
    node_start = time()
     train_NODE(
        NodeArgs(
            extrapolate = false,
            use_gpu = false,
            latent_dim = ae_args.latent_dim,  # match AE latent dim
        );
        ae = ae,
        ae_ps = ae_ps,
        ae_st = ae_st,
        normalizer = normalizer,
        ae_args = ae_args,
     )
    @info "NODE training complete" elapsed_min=round((time()-node_start)/60; digits=1)

    @info "Pipeline complete" total_min=round((time()-total_start)/60; digits=1)
   
end

main()

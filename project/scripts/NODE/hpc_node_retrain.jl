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

    device = get_device()

    # ── Step 1: Train (or load) the Autoencoder ──
       # Load the initial NODE checkpoint (trained on TL1 AE)
    ae_checkpoint_tl1 = joinpath("", "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2")
    ae_bundle_tl1, ae_args_tl1 = load_trained_AE(ae_checkpoint_tl1; device=device, return_params=true)
    ae_args_tl1.full_data_path = joinpath(root_path, ae_args_tl1.full_data_path)
    normalizer_tl1 = load_normalizer(ae_checkpoint_tl1)

    node_path = train_NODE(
        NodeArgs(
            extrapolate=false, use_gpu=false,
            latent_dim=ae_args_tl1.latent_dim, retrain=false,
        );
        ae_bundle=ae_bundle_tl1,
        normalizer=normalizer_tl1, ae_args=ae_args_tl1,
    )
    @info "Initial NODE trained" node_path

    # ── Step 2: Autoencoder has been retrained ──
    ae_checkpoint = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL2_E300_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2")
    
    # Load the trained AE into memory (no need to reload later)
    ae_bundle, ae_args = load_trained_AE(ae_checkpoint; device=device, return_params=true)
    normalizer = load_normalizer(ae_checkpoint)
    node_retrain_start = time()

    node_retrain_path = train_NODE(
        NodeArgs(
            extrapolate = false,
            latent_dim = ae_args.latent_dim,
            η = 0.01,              # lower LR for fine-tuning
            maxiters = 75,          # more iterations
            group_size = 30,         # keep tighter segments
            continuity_term = 100,   # stronger continuity for stability
            downsample = 300,  
            retrain = true,
            multiple_shooting = true,
            use_gpu = false, 
            node_checkpoint = node_path,
        );
        ae_bundle = ae_bundle,
        normalizer = normalizer, ae_args = ae_args,
    )
    node_retrain_elapsed = round((time() - node_retrain_start) / 60; digits=1)
    @info "NODE retraining complete" elapsed_min=node_retrain_elapsed node_path=node_retrain_path


end

main()

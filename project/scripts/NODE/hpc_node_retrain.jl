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


    device = get_device()

    # ── Step 1: Train (or load) the Autoencoder ──
    ae_checkpoint = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2")
    
    # Load the trained AE into memory (no need to reload later)
    _, _, ae, ae_ps, ae_st, ae_args = load_trained_AE(ae_checkpoint; device=device, return_params=true)
    normalizer = load_normalizer(ae_checkpoint)
    @info "AE loaded into memory"

    node_checkpoint = joinpath(root_path, "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2")

    train_NODE(
        NodeArgs(
            train_latent_path = train_latent_path,
            test_latent_path = test_latent_path,
            total_latent_path = total_latent_path,
            extrapolate = false,
            use_gpu = false,
            latent_dim = ae_args.latent_dim,
            η = 0.001,              # optionally use a lower LR for fine-tuning
            # maxiters = 500,         # additional iterations
            retrain = true,
            node_checkpoint = node_checkpoint,
        );
        # ae = ae,
        # ae_ps = ae_ps,
        # ae_st = ae_st,
        # normalizer = normalizer,
        # ae_args = ae_args,
    )
end

main()

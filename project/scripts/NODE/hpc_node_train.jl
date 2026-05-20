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


    # train_latent_path = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_train.jld2")
    # test_latent_path  = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_test.jld2")
    # total_latent_path = joinpath(root_path, "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2")


    device = get_device()

    # ── Step 1: Train (or load) the Autoencoder ──
    # ae_checkpoint = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E1000-100div_100curl_ground_truth/checkpoint.jld2")
    ae_checkpoint = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2")
    
    # Load the trained AE into memory (no need to reload later)
    ae_bundle, ae_args = load_trained_AE(ae_checkpoint; device=device, return_params=true)
    ae_args.t_training = 16.603

    normalizer = load_normalizer(ae_checkpoint)
    @info "AE loaded into memory"

    # ── Step 2: Train NODE using in-memory AE ──

     train_NODE(
            NodeArgs(
                maxiters=250,
                extrapolate = false,
                multiple_shooting=true,
                use_gpu = false,
                latent_dim = ae_args.latent_dim,  # match AE latent dim
            );
            ae_bundle = ae_bundle,
            normalizer = normalizer,
            ae_args = ae_args,
        )

    # for train_iters in [2, 20, 100, 250]
    #     train_NODE(
    #         NodeArgs(
    #             train_latent_path = train_latent_path,
    #             test_latent_path = test_latent_path,
    #             total_latent_path = total_latent_path,
    #             maxiters=train_iters,
    #             extrapolate = false,
    #             multiple_shooting=true,
    #             use_gpu = false,
    #             latent_dim = ae_args.latent_dim,  # match AE latent dim
    #         );
    #         ae_bundle = ae_bundle,
    #         normalizer = normalizer,
    #         ae_args = ae_args,
    #     )
    # end
end

main()

#!/usr/bin/env julia

using Thesis
using JLD2

function main()
    root_path = ""
    if is_hpc()
        root_path = "/scratch/mfbclaeys"
        @info "Starting HPC AE → NODE pipeline"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  Hostname: $(gethostname())"
    end

    device = get_device()

    # ── Step 1: Train (or load) the Autoencoder ──
    ae_checkpoint = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2")
    
    # Load the trained AE into memory (no need to reload later)
    _, _, ae, ae_ps, ae_st, ae_args = load_trained_AE(ae_checkpoint; device=device, return_params=true)
    normalizer = load_normalizer(ae_checkpoint)
    @info "AE loaded into memory"

    # ── Step 2: Train NODE using in-memory AE ──
    train_NODE(
        NodeArgs(
            extrapolate = false,
            use_gpu = true,
            latent_dim = ae_args.latent_dim,  # match AE latent dim
        );
        ae = ae,
        ae_ps = ae_ps,
        ae_st = ae_st,
        normalizer = normalizer,
        ae_args = ae_args,
        device = device,
    )
end

main()

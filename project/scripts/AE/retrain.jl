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

    # sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)


    node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
    AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
    # aenode = AENODE(AE_path, node_path)
    checkpoint = JLD2.load(AE_path)
    args_dict = checkpoint["args"]
    ae_args = LuxArgs(; args_dict...)



    # @show aenode.ae_args.λdiv, aenode.ae_args.λcurl, aenode.ae_args.λstrain
    # ---- simulation running & AENODE using to integrate
    # ---- retrain criteria is triggered
    tl_path = "data/datasets/RE2500/2e8/U_128_full.jld2"
    # retraindata = simdata = load_simdata(tl_path)
    retrain_crit = true
    # test = LuxArgs(aenode.ae_args)
    if retrain_crit
        ae_args.epochs = 20
        ae_args.retrain = true
        ae_args.checkpoint_path = AE_path
        # ae_args.full_data_path = tl_path
        # ae_args.t_training = retraindata.time[end] * 0.8
        ae_args.test_downsample = 100
        ae_args.test_loss = true
        
        train_AE(ae_args)
    end


end

main()

#!/usr/bin/env julia

using Thesis
using JLD2

function main()
    root_path = is_hpc() ? "/scratch/mfbclaeys" : ""

    device = get_device()

    # Load the initial NODE checkpoint (trained on TL1 AE)
    ae_checkpoint_tl1 = joinpath("", "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2")
    _, _, ae_tl1, ae_ps_tl1, ae_st_tl1, ae_args_tl1 = load_trained_AE(ae_checkpoint_tl1; device=device, return_params=true)
    ae_args_tl1.full_data_path = joinpath(root_path, ae_args_tl1.full_data_path)
    normalizer_tl1 = load_normalizer(ae_checkpoint_tl1)

    node_path = train_NODE(
        NodeArgs(
            extrapolate=false, use_gpu=false,
            latent_dim=ae_args_tl1.latent_dim, retrain=false,
        );
        ae=ae_tl1, ae_ps=ae_ps_tl1, ae_st=ae_st_tl1,
        normalizer=normalizer_tl1, ae_args=ae_args_tl1,
    )
    @info "Initial NODE trained" node_path

    # Load the retrained AE (TL2)
    ae_checkpoint_tl2 = joinpath("", "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL2_E300_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2")
    _, _, ae, ae_ps, ae_st, ae_args = load_trained_AE(ae_checkpoint_tl2; device=device, return_params=true)
    ae_args.full_data_path = joinpath(root_path, ae_args.full_data_path)

    normalizer = load_normalizer(ae_checkpoint_tl2)

    # Parameter grid
    ηs = [0.001, 0.002, 0.005, 0.01]
    group_sizes = [10, 15, 20, 30]
    continuity_terms = [100, 200, 400]
    downsamples = [300, 500, 750]
    maxiters_options = [50, 100, 150]

    results = []

    total_configs = length(ηs) * length(group_sizes) * length(continuity_terms) * length(downsamples) * length(maxiters_options)
    @info "Starting sweep" total_configs

    run_idx = 0
    for η in ηs, gs in group_sizes, ct in continuity_terms, ds in downsamples, mi in maxiters_options
        run_idx += 1
        @info "Run $run_idx/$total_configs" η group_size=gs continuity_term=ct downsample=ds maxiters=mi

        t_start = time()
        try
            retrain_path = train_NODE(
                NodeArgs(
                    extrapolate=false,
                    latent_dim=ae_args.latent_dim,
                    η=η,
                    maxiters=mi,
                    group_size=gs,
                    continuity_term=ct,
                    downsample=ds,
                    retrain=true,
                    multiple_shooting=true,
                    use_gpu=false,
                    node_checkpoint=node_path,
                );
                ae=ae, ae_ps=ae_ps, ae_st=ae_st,
                normalizer=normalizer, ae_args=ae_args,
            )
            elapsed = time() - t_start

            node_retrained, _ = load_node(retrain_path; verbose=false)
            z, t, _, z0 = Thesis.get_latent_vectors(ae, ae_ps, ae_st, normalizer, ae_args; device=cpu_device(), downsample=ds)
            eval = eval_node_loss(node_retrained, z, z0)

            entry = (;
                run=run_idx, η, group_size=gs, continuity_term=ct,
                downsample=ds, maxiters=mi,
                mae=eval.mae, rmse=eval.rmse, rel_l2=eval.rel_l2,
                elapsed_s=round(elapsed; digits=1),
                path=retrain_path,
            )
            push!(results, entry)
            @info "Run $run_idx complete" entry.mae entry.rmse entry.rel_l2 entry.elapsed_s

        catch e
            elapsed = time() - t_start
            @warn "Run $run_idx failed" η gs ct ds mi exception=(e, catch_backtrace())
            push!(results, (;
                run=run_idx, η, group_size=gs, continuity_term=ct,
                downsample=ds, maxiters=mi,
                mae=NaN, rmse=NaN, rel_l2=NaN,
                elapsed_s=round(elapsed; digits=1),
                path="FAILED",
            ))
        end
    end

    # Sort by MAE and print top 10
    valid = filter(r -> !isnan(r.mae), results)
    sorted = sort(valid; by=r -> r.mae)

    println("\n" * "="^90)
    println("TOP 10 CONFIGURATIONS (by MAE)")
    println("="^90)
    for (i, r) in enumerate(sorted[1:min(10, length(sorted))])
        println("  #$i  MAE=$(round(r.mae; digits=6))  RMSE=$(round(r.rmse; digits=6))  " *
                "rel_L2=$(round(r.rel_l2; digits=4))  " *
                "η=$(r.η)  gs=$(r.group_size)  ct=$(r.continuity_term)  " *
                "ds=$(r.downsample)  iters=$(r.maxiters)  time=$(r.elapsed_s)s")
    end

    # Also find best accuracy-per-second
    sorted_efficiency = sort(valid; by=r -> r.mae * r.elapsed_s)
    println("\n" * "="^90)
    println("TOP 10 CONFIGURATIONS (by MAE × time, accuracy/speed tradeoff)")
    println("="^90)
    for (i, r) in enumerate(sorted_efficiency[1:min(10, length(sorted_efficiency))])
        println("  #$i  MAE=$(round(r.mae; digits=6))  time=$(r.elapsed_s)s  " *
                "score=$(round(r.mae * r.elapsed_s; digits=4))  " *
                "η=$(r.η)  gs=$(r.group_size)  ct=$(r.continuity_term)  " *
                "ds=$(r.downsample)  iters=$(r.maxiters)")
    end

    # Save all results
    sweep_path = joinpath("data", "NODE_models", "retrain_sweep_results.jld2")
    @save sweep_path results
    @info "Sweep results saved" sweep_path n_total=length(results) n_valid=length(valid)
end

main()

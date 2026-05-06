#!/usr/bin/env julia
ENV["THESIS_HPC"] = "true"
ENV["THESIS_USE_CUDA"] = "true"

using Thesis
using JLD2
using Statistics
using LinearAlgebra
using Printf
using Dates

include(joinpath(@__DIR__, "_eval_io.jl"))

"""
    build_configs()

Return a Vector of NamedTuples, each holding the LuxArgs overrides for one run.
Edit this list to define the sweep. Common fields kept fixed across runs are
applied in `main()` via `LuxArgs(; common..., cfg...)`.
"""
function build_configs()
    return [
        # --- loss-weight / latent-dim / lr / loss-type ablations ---
        (tag="baseline",   λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:L1),
        (tag="div_zero",   λdiv=0.0,    λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:L1),
        (tag="div_low",    λdiv=100.0,  λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:L1),
        (tag="div_high",   λdiv=10000.0,λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:L1),
        (tag="curl_zero",  λdiv=1000.0, λcurl=0.0,    latent_dim=16, η=1e-3,  loss=:L1),
        (tag="curl_high",  λdiv=1000.0, λcurl=1000.0, latent_dim=16, η=1e-3,  loss=:L1),
        (tag="z8",         λdiv=1000.0, λcurl=100.0,  latent_dim=8,  η=1e-3,  loss=:L1),
        (tag="z32",        λdiv=1000.0, λcurl=100.0,  latent_dim=32, η=1e-3,  loss=:L1),
        (tag="lr_low",     λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=5e-4,  loss=:L1),
        (tag="loss_L2",    λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:L2),
        (tag="loss_charb", λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3,  loss=:charb),

        # --- architecture ablations (n_conv / n_dense / C_base) ---
        # n_conv: spatial size after encoder = 256 / 2^n_conv. n_conv≤8 to keep ≥1×1.
        (tag="nc4",        λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, n_conv=4),
        (tag="nc5",        λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, n_conv=5),
        (tag="nc7",        λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, n_conv=7),
        # n_dense: number of dense layers in bottleneck (flatten + intermediates + final-to-latent).
        (tag="nd1",        λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, n_dense=1),
        (tag="nd3",        λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, n_dense=3),
        # C_base: first conv-block channel count; subsequent blocks double.
        (tag="cbase4",     λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, C_base=4),
        (tag="cbase16",    λdiv=1000.0, λcurl=100.0,  latent_dim=16, η=1e-3, loss=:L1, C_base=16),
    ]
end

cfg_dirname(i, cfg) = "cfg$(lpad(i, 3, '0'))_$(cfg.tag)"

function run_one_config(i::Int, cfg::NamedTuple, common::NamedTuple, sweep_dir::String,
                        simdata_ram, train_data_path::String)
    cfg_dir = joinpath(sweep_dir, cfg_dirname(i, cfg))
    !ispath(cfg_dir) && mkpath(cfg_dir)

    # strip the :tag field; it's metadata, not a LuxArgs key
    cfg_kwargs = Base.structdiff(cfg, NamedTuple{(:tag,)})

    args = LuxArgs(; common...,
        save_path = cfg_dir,
        full_data_path = train_data_path,
        simdata_ram = simdata_ram,
        cfg_kwargs...,
    )

    @info "[$i] Training $(cfg.tag)" λdiv=args.λdiv λcurl=args.λcurl latent_dim=args.latent_dim η=args.η loss=args.loss epochs=args.epochs
    t0 = time()
    _, ckpt_path = train_AE(args; return_path=true)
    @info "[$i] Trained $(cfg.tag)" elapsed_s=round(time() - t0; digits=1) ckpt_path
    return ckpt_path
end

function main()
    root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
    if is_hpc()
        @info "Starting HPC AE sweep job"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  Hostname: $(gethostname())"
    end

    # ----- USER CONFIG (edit per run) -----
    sweep_name      = "sweep_$(Dates.format(now(), "yyyy-mm-dd_HHMM"))"
    sweep_dir       = joinpath(root_path, "data/Lux_models", sweep_name)
    train_data_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_full.jld2")
    test_data_path  = train_data_path  # same file; train uses time-split, eval uses all snapshots
    eval_batch_size = 16

    # common training defaults applied to every config (each cfg overrides as needed)
    common = (
        epochs          = 300,
        train_downsample= 500,
        test_downsample = 200,
        t_training      = 30.0,
        batch_size      = 16,
        normalize       = true,
        seed            = 42,
        test_loss       = false,    # skip per-epoch full-test pass; the sweep's own eval covers it
    )

    # pass/fail thresholds
    relL2_vec_max         = 0.10f0
    max_div_abs           = 0.05f0
    max_div_mult          = 5.0f0
    relL2_omega_wake_max  = 0.20f0
    # ---------------------------------------

    !ispath(sweep_dir) && mkpath(sweep_dir)
    !ispath(train_data_path) && error("train data not found: $train_data_path")
    !ispath(test_data_path) && error("test data not found: $test_data_path")

    @info "Sweep dir:        $sweep_dir"
    @info "Train data:       $train_data_path"
    @info "Test data:        $test_data_path"

    @info "Preloading simdata to RAM (shared across training runs)"
    simdata_ram = load_simdata(train_data_path)
    @info "Loaded simdata" snapshots=size(simdata_ram.u, 4)

    configs = build_configs()
    @info "Sweep size: $(length(configs)) configs"

    # ---- TRAINING ----
    trained = Tuple{String,String,NamedTuple}[]  # (cfg_name, ckpt_path, cfg)
    for (i, cfg) in enumerate(configs)
        try
            ckpt_path = run_one_config(i, cfg, common, sweep_dir, simdata_ram, train_data_path)
            push!(trained, (cfg_dirname(i, cfg), ckpt_path, cfg))
        catch e
            @error "Training failed for config $i ($(cfg.tag))" exception=(e, catch_backtrace())
        end
    end

    isempty(trained) && (@error "No configs trained successfully"; return)
    @info "Training complete: $(length(trained)) / $(length(configs)) succeeded"

    # ---- EVALUATION ----
    @info "Loading test simdata for evaluation"
    test_simdata = test_data_path == train_data_path ? simdata_ram : load_simdata(test_data_path)
    if test_data_path != train_data_path
        preprocess_data!(test_simdata; verbose=true)
    else
        # simdata_ram was passed through train_AE which calls preprocess_data! in get_data;
        # confirm it has been clipped (no ghost cells -> H is power of 2)
        if !ispow2(size(test_simdata.u, 1))
            preprocess_data!(test_simdata; verbose=true)
        end
    end

    device = get_device()

    @info "Computing baseline divergence on test data"
    baseline = compute_baseline_divergence(test_simdata, device, eval_batch_size)
    @info "Baseline divergence" max=baseline.max mean=baseline.mean

    rows = NamedTuple[]
    for (cfg_name, ckpt_path, _) in trained
        @info "Evaluating $cfg_name"
        local metrics, args
        try
            metrics, args = evaluate_checkpoint(ckpt_path, test_simdata, device, eval_batch_size)
        catch e
            @error "Failed to evaluate $cfg_name" exception=(e, catch_backtrace())
            continue
        end
        status, reason = pass_fail(metrics, baseline.max;
            relL2_vec_max=relL2_vec_max,
            max_div_abs=max_div_abs,
            max_div_mult=max_div_mult,
            relL2_omega_wake_max=relL2_omega_wake_max)
        push!(rows, build_eval_row(cfg_name, metrics, args, baseline, status, reason))
        @info "  status=$status relL2_vec=$(round(metrics.relL2_vec; digits=4)) max_div=$(round(metrics.max_div; digits=4)) relL2_omega_wake=$(round(metrics.relL2_omega_wake; digits=4))"
    end

    isempty(rows) && (@error "No evaluations completed"; return)

    header = vcat(:checkpoint, collect(HPARAM_COLS), collect(METRIC_COLS),
                  :baseline_max_div, :baseline_mean_div, collect(STATUS_COLS))
    csv_path = joinpath(sweep_dir, "evaluation_results.csv")
    write_csv(csv_path, rows, header)
    @info "Wrote $csv_path"

    md_path = joinpath(sweep_dir, "evaluation_summary.md")
    write_summary(md_path, rows, baseline, (;
        relL2_vec_max, max_div_abs, max_div_mult, relL2_omega_wake_max))
    @info "Wrote $md_path"
end

main()

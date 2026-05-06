#!/usr/bin/env julia
ENV["THESIS_HPC"] = "true"
ENV["THESIS_USE_CUDA"] = "true"

using Thesis
using JLD2
using Statistics
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "_eval_io.jl"))

function main()
    root_path = is_hpc() ? "/scratch/mfbclaeys" : ""
    if is_hpc()
        @info "Starting HPC AE evaluation job"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  Hostname: $(gethostname())"
    end

    # ----- USER CONFIG (edit per run) -----
    ckpt_dir   = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/sweep_001")
    test_path  = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_full.jld2")
    output_dir = ckpt_dir
    eval_batch_size = 16

    relL2_vec_max         = 0.10f0
    max_div_abs           = 0.05f0
    max_div_mult          = 5.0f0
    relL2_omega_wake_max  = 0.20f0
    # ---------------------------------------

    !ispath(output_dir) && mkpath(output_dir)
    isfile(test_path) || error("test dataset not found: $test_path")

    @info "Checkpoint dir: $ckpt_dir"
    @info "Test data:      $test_path"
    @info "Output dir:     $output_dir"

    device = get_device()
    @info "Device: $device"

    @info "Loading test simdata"
    simdata = load_simdata(test_path)
    preprocess_data!(simdata; verbose=true)
    @info "Test snapshots: $(size(simdata.u, 4))"

    @info "Computing baseline divergence on input data"
    baseline = compute_baseline_divergence(simdata, device, eval_batch_size)
    @info "Baseline divergence" max=baseline.max mean=baseline.mean

    ckpts = find_checkpoints(ckpt_dir)
    isempty(ckpts) && error("no checkpoints found under $ckpt_dir")
    @info "Found $(length(ckpts)) checkpoints"

    rows = NamedTuple[]
    for (name, path) in ckpts
        @info "Evaluating $name"
        local metrics, args
        try
            metrics, args = evaluate_checkpoint(path, simdata, device, eval_batch_size)
        catch e
            @error "Failed to evaluate $name" exception=(e, catch_backtrace())
            continue
        end
        status, reason = pass_fail(metrics, baseline.max;
            relL2_vec_max=relL2_vec_max,
            max_div_abs=max_div_abs,
            max_div_mult=max_div_mult,
            relL2_omega_wake_max=relL2_omega_wake_max)
        push!(rows, build_eval_row(name, metrics, args, baseline, status, reason))
        @info "  status=$status relL2_vec=$(round(metrics.relL2_vec; digits=4)) max_div=$(round(metrics.max_div; digits=4)) relL2_omega_wake=$(round(metrics.relL2_omega_wake; digits=4))"
    end

    isempty(rows) && (@error "No checkpoints evaluated successfully"; return)

    header = vcat(:checkpoint, collect(HPARAM_COLS), collect(METRIC_COLS),
                  :baseline_max_div, :baseline_mean_div, collect(STATUS_COLS))
    csv_path = joinpath(output_dir, "evaluation_results.csv")
    write_csv(csv_path, rows, header)
    @info "Wrote $csv_path"

    md_path = joinpath(output_dir, "evaluation_summary.md")
    write_summary(md_path, rows, baseline, (;
        relL2_vec_max, max_div_abs, max_div_mult, relL2_omega_wake_max))
    @info "Wrote $md_path"
end

main()

#!/usr/bin/env julia

ENV["THESIS_HPC"] = "true"
ENV["THESIS_USE_CUDA"] = "true"

using Thesis
using Statistics
using JLD2
using Dates
using Printf

const LAMBDA_DIV   = (0.0, 100.0, 10000.0)
const LAMBDA_CURL  = (0.0, 100.0, 10000.0)
const LATENT_DIM   = (8, 16, 32)
const N_CONV       = (4, 5, 6)
const N_DENSE      = (1, 2, 3)
const EPOCHS       = 200

function build_grid()
    configs = NamedTuple[]
    for λdiv in LAMBDA_DIV, λcurl in LAMBDA_CURL,
        z in LATENT_DIM, nc in N_CONV, nd in N_DENSE
        push!(configs, (λdiv=λdiv, λcurl=λcurl, latent_dim=z, n_conv=nc, n_dense=nd))
    end
    return configs
end

function evaluate_checkpoint(ckpt_path::String; simdata_ram=nothing)
    cpu = cpu_device()
    ae_bundle, args = load_trained_AE(ckpt_path; return_params=true)
    ae, ps, st = ae_bundle.ae, cpu(ae_bundle.ps), cpu(ae_bundle.st)
    normalizer = load_normalizer(ckpt_path)

    args.simdata_ram = simdata_ram
    data, loaders, _ = get_data(args.batch_size, args.full_data_path;
        n_training=args.train_downsample, n_test=args.test_downsample,
        split=args.split, t_training=args.t_training,
        verbose=false, simdata_ram=simdata_ram)
    TestData = data.TestData

    rec_sum = 0.0; div_sum = 0.0; curl_sum = 0.0; tke_sum = 0.0; n = 0
    for idx in loaders.test_loader
        x_in, x_target, μ₀ = build_batch(TestData, idx)
        x_in_f = Float32.(x_in)
        if args.normalize
            uvc = x_in_f[:, :, 1:2, :]
            uvc_norm, _ = normalize_batch(uvc; normalizer=normalizer)
            x_in_f = cat(uvc_norm, x_in_f[:, :, 3:4, :]; dims=3)
        end
        x̂_norm, _ = ae(x_in_f, ps, st)
        x̂ = Array(denormalize_batch(x̂_norm, normalizer)) .* Array(μ₀)
        x = Array(x_target)

        rec_sum  += mean(abs, x .- x̂)
        div_sum  += mean(abs, div_field(x̂; buff=1))
        curl_sum += mean(abs, curl_vectorized(x; buff=1) .- curl_vectorized(x̂; buff=1))
        tke_x = redirect_stdout(devnull) do
            kinetic_energy_dissipation(x; ν=1.0, avg=true, buff=1)
        end
        tke_x̂ = redirect_stdout(devnull) do
            kinetic_energy_dissipation(x̂; ν=1.0, avg=true, buff=1)
        end
        tke_sum += mean(abs, tke_x .- tke_x̂)
        n += 1
    end
    n = max(n, 1)
    return (recon_mae=rec_sum/n, divergence=div_sum/n,
            curl_err=curl_sum/n, tke_err=tke_sum/n)
end

function rank_normalize(v::Vector{Float64})
    # smaller is better; normalize each metric to [0,1] using min-max
    lo, hi = minimum(v), maximum(v)
    hi == lo && return zeros(length(v))
    return (v .- lo) ./ (hi - lo)
end

function write_summary_csv(rows::Vector{<:NamedTuple}, csv_path::String)
    cols = [:tag, :λdiv, :λcurl, :latent_dim, :n_conv, :n_dense,
            :recon_mae, :divergence, :curl_err, :tke_err, :score, :rank]
    open(csv_path, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join((string(getfield(r, c)) for c in cols), ","))
        end
    end
    @info "Summary CSV written to $csv_path"
end

function summarize_and_rank(results::Vector{<:NamedTuple}, csv_path::String)
    finite = filter(r -> all(isfinite, (r.recon_mae, r.divergence, r.curl_err, r.tke_err)), results)
    if isempty(finite)
        @warn "No finite results to rank"
        return
    end

    rec  = rank_normalize([Float64(r.recon_mae)  for r in finite])
    dvg  = rank_normalize([Float64(r.divergence) for r in finite])
    crl  = rank_normalize([Float64(r.curl_err)   for r in finite])
    tke  = rank_normalize([Float64(r.tke_err)    for r in finite])
    score = (rec .+ dvg .+ crl .+ tke) ./ 4

    order = sortperm(score)
    ranked = [merge(finite[i], (score=score[i], rank=findfirst(==(i), order))) for i in eachindex(finite)]
    sort!(ranked, by=r->r.rank)
    write_summary_csv(ranked, csv_path)
end

function tag_for(cfg::NamedTuple)
    return @sprintf("d%g_c%g_z%d_nc%d_nd%d",
        cfg.λdiv, cfg.λcurl, cfg.latent_dim, cfg.n_conv, cfg.n_dense)
end

function main()
    root_path = ""
    if is_hpc()
        root_path = "/scratch/mfbclaeys"
        @info "Starting HPC AE tuning job"
        @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
        @info "  Hostname: $(gethostname())"
    end

    data_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_full.jld2")
    @info "Loading training data from: $data_path"

    simdata = load_simdata(data_path)

    grid = build_grid()
    @info "Grid search over $(length(grid)) configurations"

    timestamp = Dates.format(now(), "udd-HHMM")
    out_dir = joinpath("data/Lux_models", "tune_$(timestamp)")
    !ispath(out_dir) && mkpath(out_dir)
    csv_path = joinpath(out_dir, "tune_summary.csv")

    results = NamedTuple[]
    for (i, cfg) in enumerate(grid)
        tag = tag_for(cfg)
        @info "[$i/$(length(grid))] training: $tag"
        try
            _, ckpt_path = train_AE(
                LuxArgs(
                    epochs=EPOCHS,
                    λdiv=Float64(cfg.λdiv),
                    λcurl=Float64(cfg.λcurl),
                    latent_dim=cfg.latent_dim,
                    n_conv=cfg.n_conv,
                    n_dense=cfg.n_dense,
                    train_downsample=300,
                    test_downsample=300,
                    t_training=25,
                    full_data_path=data_path,
                    simdata_ram=simdata,
                ); return_path=true
            )
            metrics = evaluate_checkpoint(ckpt_path; simdata_ram=simdata)
            push!(results, (tag=tag, λdiv=cfg.λdiv, λcurl=cfg.λcurl,
                            latent_dim=cfg.latent_dim, n_conv=cfg.n_conv, n_dense=cfg.n_dense,
                            metrics...))
            @info "  → recon=$(metrics.recon_mae) div=$(metrics.divergence) curl=$(metrics.curl_err) tke=$(metrics.tke_err)"
        catch e
            @error "Run $tag failed" exception=(e, catch_backtrace())
        end
        # incremental save in case the job dies mid-grid
        !isempty(results) && summarize_and_rank(results, csv_path)
    end

    @info "Tuning complete. $(length(results))/$(length(grid)) runs succeeded."
end

main()

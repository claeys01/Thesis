# CSV / Markdown / Spearman helpers shared by hpc_evaluate.jl and hpc_sweep.jl.
# Driver-side code: lives in scripts/, not src/.

function _ranks(x::AbstractVector)
    n = length(x)
    p = sortperm(x)
    r = zeros(Float64, n)
    @inbounds for i in 1:n
        r[p[i]] = i
    end
    return r
end

function spearman(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch())
    n < 3 && return NaN
    rx = _ranks(Float64.(x)); ry = _ranks(Float64.(y))
    (std(rx) == 0 || std(ry) == 0) && return NaN
    return cor(rx, ry)
end

function find_checkpoints(parent::AbstractString)
    isdir(parent) || error("checkpoint directory not found: $parent")
    out = Tuple{String,String}[]
    for entry in sort(readdir(parent))
        sub = joinpath(parent, entry)
        ckpt = joinpath(sub, "checkpoint.jld2")
        isdir(sub) && isfile(ckpt) && push!(out, (entry, ckpt))
        if isdir(sub)
            for entry2 in sort(readdir(sub))
                sub2 = joinpath(sub, entry2)
                ckpt2 = joinpath(sub2, "checkpoint.jld2")
                isdir(sub2) && isfile(ckpt2) && push!(out, (joinpath(entry, entry2), ckpt2))
            end
        end
    end
    direct = joinpath(parent, "checkpoint.jld2")
    isfile(direct) && push!(out, (basename(parent), direct))
    return out
end

_csv_cell(x::AbstractFloat) = isnan(x) ? "NaN" : @sprintf("%.6g", x)
_csv_cell(x::Symbol) = string(x)
_csv_cell(x::Bool) = x ? "true" : "false"
_csv_cell(x) = string(x)

function write_csv(path::String, rows::Vector{<:NamedTuple}, header::Vector{Symbol})
    open(path, "w") do io
        println(io, join(string.(header), ","))
        for row in rows
            cells = [_csv_cell(getfield(row, k)) for k in header]
            println(io, join(cells, ","))
        end
    end
    return path
end

function format_md_row(row::NamedTuple, cols::Tuple)
    pieces = String[]
    for k in cols
        v = getfield(row, k)
        if v isa AbstractFloat
            push!(pieces, "$(k)=$(round(v; sigdigits=4))")
        else
            push!(pieces, "$(k)=$(v)")
        end
    end
    return join(pieces, ", ")
end

function write_summary(path::String, rows::Vector{<:NamedTuple}, baseline::NamedTuple, args_kwargs::NamedTuple)
    pass = filter(r -> r.status == "PASS", rows)
    fail = filter(r -> r.status == "FAIL", rows)
    sort!(pass, by=r -> r.relL2_vec)
    open(path, "w") do io
        println(io, "# AE evaluation summary")
        println(io)
        println(io, "Baseline (input data) divergence: max=$(round(baseline.max; digits=5)), mean=$(round(baseline.mean; digits=5))")
        println(io, "Pass thresholds: relL2_vec<$(args_kwargs.relL2_vec_max), max_div<max($(args_kwargs.max_div_abs), $(args_kwargs.max_div_mult)×baseline_max_div)=$(round(max(args_kwargs.max_div_abs, args_kwargs.max_div_mult * baseline.max); digits=5)), relL2_omega_wake<$(args_kwargs.relL2_omega_wake_max)")
        println(io)
        println(io, "## Top 5 PASS runs (ranked by relL2_vec ascending)")
        println(io)
        if isempty(pass)
            println(io, "_No runs passed all constraints._")
        else
            for (i, r) in enumerate(pass[1:min(5, length(pass))])
                println(io, "$(i). **$(r.checkpoint)**")
                println(io, "   - hparams: $(format_md_row(r, HPARAM_COLS))")
                println(io, "   - metrics: $(format_md_row(r, METRIC_COLS))")
            end
        end
        println(io)
        println(io, "## FAIL runs")
        println(io)
        if isempty(fail)
            println(io, "_None._")
        else
            for r in fail
                println(io, "- **$(r.checkpoint)** — $(r.reason)")
            end
        end
        println(io)
        println(io, "## Spearman correlations vs relL2_vec")
        println(io)
        if length(rows) >= 3
            valid = filter(r -> !isnan(r.relL2_vec), rows)
            y = Float64[Float64(r.relL2_vec) for r in valid]
            for k in (:λdiv, :λcurl, :latent_dim, :η, :batch_size)
                xs = Float64[Float64(getfield(r, k)) for r in valid]
                ρ = spearman(xs, y)
                println(io, "- $(k): ρ = $(isnan(ρ) ? "NaN" : round(ρ; digits=3))")
            end
        else
            println(io, "_Need ≥3 runs for Spearman correlations._")
        end
    end
    return path
end

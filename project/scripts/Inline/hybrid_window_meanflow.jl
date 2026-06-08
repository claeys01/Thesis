using Thesis
using WaterLily
using WaterLily: MeanFlow
using JLD2
using Statistics
using Plots

base_dir = "data/saved_models/inline_runs_hpc/base"

# mode_log gives the "Hybrid" spans (sim-time, tU/L); the whole-run mean flows are
# loaded only to contrast them against the hybrid-only ones computed below.
hs_path = joinpath(base_dir, "hybrid_state.jld2")
@load hs_path sim_meanflow ref_meanflow mode_log

hybrid_path = joinpath(base_dir, "U_hybrid_inline.jld2")
ref_path    = joinpath(base_dir, "U_ref_inline.jld2")

hybrid_spans = [(l.t_start, l.t_end) for l in mode_log if l.mode == "Hybrid"]
isempty(hybrid_spans) && error("No \"Hybrid\" spans in mode_log — nothing to average.")

# Each field file stores, per saved step i: u/$i (the raw flow.u, an (Nx,Ny,2) array
# with ghost cells, last axis the velocity component), t/$i (sim time, tU/L) and n
# (record count). Records from every phase (warmup, hybrid, cutoff) are present; the
# hybrid/reference pairs are written in lock-step, so a single index refers to the
# same physical instant in both files. The files are multi-GB, so only the per-record
# scalar times are read up front; the fields themselves are streamed one at a time.
function load_times(path)
    jldopen(path, "r") do f
        n = f["n"]
        return Float32[f["t/$i"] for i in 1:n]
    end
end

field_size(path) = jldopen(f -> size(f["u/1"]), path, "r")

# Trapezoidal time weights over sorted instants: wᵢ is the time interval each
# snapshot represents, so the average is a genuine time-average ∫u dt / ∫dt. A plain
# per-snapshot mean would over-weight the densely sampled CFD stretches relative to
# the coarser rollout snapshots that share the same window.
function trapz_weights(t::AbstractVector)
    n = length(t)
    n == 1 && return ones(eltype(t), 1)
    w = zeros(eltype(t), n)
    w[1]   = (t[2] - t[1]) / 2
    w[end] = (t[end] - t[end-1]) / 2
    for i in 2:n-1
        w[i] = (t[i+1] - t[i-1]) / 2
    end
    return w
end

# (record index, time weight) pairs covering every hybrid window. Each window is
# weighted on its own sorted timestamps, so the cutoff/retrain gaps between windows
# never bias the average. Repeated instants are dropped to keep the weights well-posed.
function window_weights(times, spans)
    idxw = Tuple{Int,Float64}[]
    W = 0.0
    for (t0, t1) in spans
        idx = findall(t -> t0 <= t <= t1, times)
        length(idx) < 2 && continue
        idx = idx[sortperm(times[idx])]
        keep = trues(length(idx))
        for j in 2:length(idx)
            keep[j] = times[idx[j]] > times[idx[j-1]]
        end
        idx = idx[keep]
        tw = trapz_weights(times[idx])
        for (w, k) in zip(tw, idx)
            push!(idxw, (k, Float64(w)))
        end
        W += sum(tw)
    end
    return idxw, W
end

# Time-averaged ⟨u⟩ and ⟨u⊗u⟩ over the selected records, returned as a MeanFlow so the
# existing meanflow_errors / plot_* helpers apply unchanged. Only U and UU are filled
# (P stays zero — it is never read by those helpers). Records are streamed from disk
# one at a time so the multi-GB files are never held in memory.
function accumulate_meanflow(path, idxw, W, fsize)
    Nx, Ny, D = fsize
    mf = MeanFlow((Nx - 2, Ny - 2); mem=Array, T=Float32, uu_stats=true)
    U  = zeros(Float64, Nx, Ny, D)
    UU = zeros(Float64, Nx, Ny, D, D)
    jldopen(path, "r") do f
        for (k, w) in idxw
            uk = f["u/$k"]
            @views U .+= w .* uk
            for i in 1:D, j in 1:D
                @views UU[:, :, i, j] .+= w .* (uk[:, :, i] .* uk[:, :, j])
            end
        end
    end
    mf.U  .= Float32.(U  ./ W)
    mf.UU .= Float32.(UU ./ W)
    return mf
end

# The pairs are lock-step, so the hybrid timestamps define both window membership and
# the weighting for both series.
t_hyb = load_times(hybrid_path)
idxw, W = window_weights(t_hyb, hybrid_spans)
isempty(idxw) && error("No hybrid-window snapshots with a positive time span were found.")
fsize = field_size(hybrid_path)

sim_mf_hyb = accumulate_meanflow(hybrid_path, idxw, W, fsize)
ref_mf_hyb = accumulate_meanflow(ref_path,    idxw, W, fsize)

println("\n" * "="^60)
println("HYBRID-WINDOW MEAN FLOW")
println("="^60)
println("  Hybrid spans:        $(length(hybrid_spans))")
for (k, (t0, t1)) in enumerate(hybrid_spans)
    println("    #$k  tU/L = [$(round(t0, digits=3)), $(round(t1, digits=3))]  (Δ = $(round(t1 - t0, digits=3)))")
end
println("  Snapshots averaged:  $(length(idxw))")
println("  Total hybrid time:   $(round(W, digits=3)) tU/L")

fe_hyb   = meanflow_errors(sim_mf_hyb, ref_mf_hyb)
fe_whole = meanflow_errors(sim_meanflow, ref_meanflow)

function print_field_block(label, fe)
    println("\n--- $label ---")
    println("MAE - Mean flow ⟨u⟩: $(round(fe.u.l1, digits=3)),  ⟨v⟩: $(round(fe.v.l1, digits=3))")
    println("MAE - RST ⟨u'u'⟩: $(round(fe.uu.l1, digits=3)) ,  ⟨v'v'⟩: $(round(fe.vv.l1, digits=3)),  ⟨u'v'⟩: $(round(fe.uv.l1, digits=3))")
    println("Corr coeff   - Mean flow ⟨u⟩: $(round(fe.u.ρ, digits=4)),  ⟨v⟩: $(round(fe.v.ρ, digits=4))")
    println("Corr coeff   - RST ⟨u'u'⟩: $(round(fe.uu.ρ, digits=4)),  ⟨v'v'⟩: $(round(fe.vv.ρ, digits=4)),  ⟨u'v'⟩: $(round(fe.uv.ρ, digits=4))")
end

print_field_block("Hybrid windows only", fe_hyb)
print_field_block("Whole run (for reference)", fe_whole)
println("\n" * "="^60)

meanplot = plot_meanflow_comparison(sim_mf_hyb, ref_mf_hyb;
    savedir=joinpath(base_dir, "meanflow_panels_hybrid_only"))
display(meanplot)

rst_plot = plot_rst_comparison(sim_mf_hyb, ref_mf_hyb; style=:filled,
    savedir=joinpath(base_dir, "rst_panels_hybrid_only"))
display(rst_plot)

mf_out = joinpath(base_dir, "meanflow_hybrid_only.jld2")
@save mf_out sim_mf_hyb ref_mf_hyb hybrid_spans
println("Saved hybrid-window mean flows to: $mf_out")

nothing

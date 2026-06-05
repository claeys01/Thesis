using JLD2
using Statistics
using Plots
using Thesis

# Run directory holding U_hybrid_inline.jld2 / U_ref_inline.jld2
savedir = "data/saved_models/inline_runs_hpc/base"

hybrid_path = joinpath(savedir, "U_hybrid_inline.jld2")
ref_path    = joinpath(savedir, "U_ref_inline.jld2")

# --- wake probe location (EDIT ME) ---------------------------------------
# Cylinder (see circle_shedding_biot): center at (n/4, m/2), diameter D = m/8.
# We put the probe a few diameters downstream, slightly off the centerline,
# and read the cross-stream velocity (comp 2) which oscillates at the
# vortex-shedding frequency.
x_downstream_D = 3.0   # distance behind cylinder center, in diameters
y_offset_D     = 0.5   # offset from the centerline, in diameters
component      = 1     # 1 = streamwise u, 2 = cross-stream v
# -------------------------------------------------------------------------

# Probe index from the field size (matches the simulation geometry).
Nx, Ny = jldopen(f -> size(f["u/1"]), ref_path, "r")   # incl. ghost cells
n_phys, m_phys = Nx - 2, Ny - 2
D = m_phys / 8
ix = round(Int, n_phys/4 + x_downstream_D * D) + 1     # +1 for the ghost cell
iy = round(Int, m_phys/2 + y_offset_D     * D) + 1
@info "Probe" ix iy component field_size=(Nx, Ny)

# Load the (time, velocity) series at the probe.
function load_probe(path, ix, iy, ic)
    jldopen(path, "r") do f
        n = f["n"]
        t = [Float32(f["t/$i"]) for i in 1:n]
        v = [Float32(f["u/$i"][ix, iy, ic]) for i in 1:n]
        return t, v
    end
end

t_ref, v_ref = load_probe(ref_path, ix, iy, component)
t_hyb, v_hyb = load_probe(hybrid_path, ix, iy, component)

vel_sym = component == 1 ? "u" : "v"   # streamwise / cross-stream

v_plot = plot(framestyle=:box, size=(800, 400), dpi=500,
    xlabel="\$t^*\$", ylabel="\$$vel_sym\$",
    title="Wake velocity at \$($(round(x_downstream_D, digits=1))D, $(round(y_offset_D, digits=1))D)\$",
    titlefontsize=14, guidefontsize=12, tickfontsize=8, legendfontsize=8,
    foreground_color_axis = :black, foreground_color_text = :black,
    left_margin = 3Plots.mm, right_margin = 1Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 2Plots.mm,
    legend = :topright, background_color_legend = RGBA(1, 1, 1, 0.7),
    xlims = (0, t_ref[end]))
plot!(v_plot, t_ref, v_ref; label="Reference", color=:red, alpha=0.5, ls=:dashdot, lw=1.5)
plot!(v_plot, t_hyb, v_hyb; label="Hybrid", color=:blue, lw=1)
display(v_plot)

# Temporal energy spectrum E(f) = |FFT(fluctuations)|^2 of (v - mean(v)).
# Plain one-sided DFT (no FFTW dependency); samples assumed ~uniform in time.
function energy_spectrum(t, v)
    N  = length(v)
    dt = (t[end] - t[1]) / (N - 1)         # mean sampling interval
    x  = v .- mean(v)                       # fluctuations only
    nf = div(N, 2)                          # positive frequencies
    freqs = [k / (N * dt) for k in 1:nf]
    E = zeros(Float64, nf)
    for k in 1:nf
        acc = 0.0 + 0.0im
        @inbounds for n in 1:N
            acc += x[n] * cis(-2π * k * (n - 1) / N)
        end
        E[k] = abs2(acc) / N^2
    end
    return freqs, E
end

f_ref, E_ref = energy_spectrum(t_ref, v_ref)
f_hyb, E_hyb = energy_spectrum(t_hyb, v_hyb)

plt = plot(f_ref, E_ref;
    label="reference", color=:black, lw=1.5,
    xlabel="frequency f", ylabel="energy E(f)",
    left_margin = 3Plots.mm, right_margin = 1Plots.mm,
    top_margin = 1Plots.mm, bottom_margin = 4Plots.mm,
    xscale=:log10, yscale=:log10,
    title="Wake energy spectrum at probe ($ix, $iy), comp $component",
    dpi=400, size=(900, 450), framestyle=:box)
plot!(plt, f_hyb, E_hyb; label="hybrid", color=:red, lw=1.5, alpha=0.7, linestyle=:dash)

# Kolmogorov -5/3 reference slope, anchored to the reference spectrum at f_anchor.
f_anchor = 1.0   # frequency to pin the slope line to (EDIT ME)
k0 = argmin(abs.(f_ref .- f_anchor))
C  = E_ref[k0] * f_ref[k0]^(5/3)          # so C*f^(-5/3) passes through that point
slope = C .* f_ref .^ (-5/3)
plot!(plt, f_ref, slope; label="-5/3 slope", color=:blue, lw=2, linestyle=:dot)

out_path = joinpath(savedir, "wake_energy_spectrum.png")
# savefig(plt, out_path)
# @info "Saved spectrum plot to $out_path"
display(plt)

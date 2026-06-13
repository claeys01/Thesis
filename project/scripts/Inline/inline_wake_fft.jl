using JLD2
using Statistics
using Plots
using Thesis
using FFTW
using DSP
using Dierckx   # cubic-spline resampling

# Run directory holding U_hybrid_inline.jld2 / U_ref_inline.jld2
savedir = "data/saved_models/inline_runs_hpc/base"
# savedir = "data/inline_runs/2026-06-09_00-26"

hybrid_path = joinpath(savedir, "U_hybrid_inline.jld2")
ref_path    = joinpath(savedir, "U_ref_inline.jld2")



# --- wake probe location (EDIT ME) ---------------------------------------
# Cylinder (see circle_shedding_biot): center at (n/4, m/2), diameter D = m/8.
# We put the probe a few diameters downstream, slightly off the centerline,
# and read the cross-stream velocity (comp 2) which oscillates at the
# vortex-shedding frequency.
x_downstream_D = 2.0   # distance behind cylinder center, in diameters
y_offset_D     = 0.5  # offset from the centerline, in diameters
component      = 2     # 1 = streamwise u, 2 = cross-stream v
n_segments     = 5     # Welch: number of averaged segments (more → smoother, coarser in f)
overlap        = 0.5   # fractional overlap between segments (0.5 is standard)
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



# Native step distributions for both runs. The two runs are independent
# adaptive solves, so they have no reason to share a length — resampling onto
# a common grid is exactly what removes that requirement, hence no size assert.
# drop non-monotone / duplicate samples (precision collapse or boundary dupes)
function dedupe_time(t, v)
    keep = [true; diff(t) .> 0]
    return t[keep], v[keep]
end
t_ref, v_ref = dedupe_time(t_ref, v_ref)
t_hyb, v_hyb = dedupe_time(t_hyb, v_hyb)

p = plot(t_ref)
plot!(t_hyb)
# display(p)

dt_ref = diff(t_ref)
dt_hyb = diff(t_hyb)
dt_min = min(minimum(dt_ref), minimum(dt_hyb))
dt_max = max(maximum(dt_ref), maximum(dt_hyb))
dt_mean = mean(append!(dt_ref, dt_hyb))
dt_median = median(append!(dt_ref, dt_hyb))
@show dt_min dt_max dt_max / dt_min dt_mean dt_median

# Resample both signals onto a common uniform grid spaced at dt_min 
Δt      = dt_min

t_start = max(t_ref[1], t_hyb[1])
t_end   = min(t_ref[end], t_hyb[end])
t_grid  = collect(t_start:Δt:t_end)

# Cubic spline interpolation. Linear interp is a triangular (sinc²) low-pass
# that attenuates the high frequencies and leaks broadband noise; for a smooth
# probe/force signal a cubic spline preserves the spectrum far better.
t_ref_samp = t_grid
t_hyb_samp = t_grid
v_ref_samp = Spline1D(t_ref, v_ref; k=3)(t_grid)
v_hyb_samp = Spline1D(t_hyb, v_hyb; k=3)(t_grid)

# new_p = plot(t_ref_samp, v_ref_samp; label = "ref (resampled)", dpi=300)
# plot!(new_p, t_hyb_samp, v_hyb_samp; label = "hyb (resampled)")
# display(new_p)

# t_train, v_train = load_probe(training_data, ix, iy, component)

# --- Welch PSD --------------------------------------------------------------
# Split the signal into `nseg` overlapping Hann windows, FFT each, average the
# power. This is the textbook Welch estimate (Welch 1967); rfft from FFTW,
# hanning from DSP.
function welch_psd(x, fs; nseg=5, overlap=0.5)
    N = length(x)
    # segment length that fits exactly `nseg` windows with the given overlap:
    # N = L + (nseg-1)*step, step = (1-overlap)*L
    L    = floor(Int, N / (1 + (nseg - 1) * (1 - overlap)))
    step = floor(Int, (1 - overlap) * L)
    win  = hanning(L)
    U    = sum(abs2, win) / L            # window power, for unbiased scaling

    nf  = div(L, 2) + 1
    psd = zeros(Float64, nf)
    for k in 0:(nseg - 1)
        seg = @view x[k*step + 1 : k*step + L]
        seg = (seg .- mean(seg)) .* win  # detrend then apply window
        psd .+= abs2.(rfft(seg))
    end
    psd ./= nseg                         # average the periodograms
    psd ./= (fs * L * U)                 # convert to a power spectral density
    psd[2:end-1] .*= 2                   # one-sided: fold negative frequencies
    f = (0:nf-1) .* (fs / L)
    return f, psd
end

fs = 1 / Δt                              # single shared sampling rate after resampling

f_ref, P_ref = welch_psd(v_ref_samp, fs; nseg=n_segments, overlap=overlap)
f_hyb, P_hyb = welch_psd(v_hyb_samp, fs; nseg=n_segments, overlap=overlap)


# --- plots ------------------------------------------------------------------
comp_label = component == 1 ? "u" : "v"

# shared publication styling: serif font, boxed frame, inward minor ticks
default(fontfamily = "Computer Modern", framestyle = :box, grid = false,
        tick_direction = :in, minorticks = true, widen = true,
        tickfontsize = 10, guidefontsize = 13, legendfontsize = 7,
        foreground_color_legend = :black, background_color_legend = :white)

ref_color = :black
hyb_color = RGB(0.85, 0.45, 0.0)

# (1) probe velocity signal
p_sig = plot(t_ref, v_ref; label = "reference", color = ref_color, lw = 1.2, ls=:dashdot, alpha = 0.5, bottom_margin=4Plots.mm, left_margin=4Plots.mm,
            ylabel = "\$$(comp_label)\$", legend = :topright, size=(800, 400), dpi=500,xlabel="\$t^*\$",)
plot!(p_sig, t_hyb, v_hyb; label = "hybrid", color = hyb_color, lw = 1.4)
# plot!(p_sig, t_train, v_train)
savefig(p_sig, joinpath(savedir, "wake_signal.pdf"))

# (2) Welch power spectrum
P_all    = vcat(P_ref[2:end], P_hyb[2:end])
ylo, yhi = minimum(filter(>(0), P_all)), maximum(P_all)
p_psd = plot(f_ref[2:end], P_ref[2:end]; label = "reference", color = ref_color,
             lw = 1.6, xscale = :log10, yscale = :log10,
             xlabel = "\$f L/U\$", ylabel = "PS(\$$(comp_label)\$)",
             legend = :bottomleft, size = (800, 400), dpi = 500,
             ylims = (ylo / 3, yhi * 2),
             grid = true, gridalpha = 0.2, gridlinewidth = 0.6,
             bottom_margin=4Plots.mm, left_margin=4Plots.mm,
             minorgrid = true, minorgridalpha = 0.05)
plot!(p_psd, f_hyb[2:end], P_hyb[2:end]; label = "hybrid", color = hyb_color, lw = 1.4)

# -5/3 inertial-subrange reference slope, spanning the full width and laid
# along the linear (inertial) decay of the spectrum
f_pos    = f_ref[2:end]
f_anchor = 2.0                                # frequency in the linear falloff to seat the line
lift     = 2.5                               # vertical shift (×); >1 lifts it off the data
ia       = argmin(abs.(f_pos .- f_anchor))
A53      = lift * P_ref[1 + ia] / f_pos[ia]^(-5/3)
f53      = exp10.(range(log10(f_pos[1]), log10(f_pos[end]); length = 100))
plot!(p_psd, f53, A53 .* f53 .^ (-5/3); label = "\$f^{-5/3}\$",
      color = :gray40, lw = 1.2, ls = :dot,)

# plot!(p_psd, f53, A53 .* f53 .^ (-3); label = "\$f^{-3}\$",
    #   color = :red, lw = 1.2, ls = :dot,)
      
savefig(p_psd, joinpath(savedir, "wake_psd.pdf"))

display(p_sig)
display(p_psd)
@info "Saved" signal=joinpath(savedir, "wake_signal.pdf") psd=joinpath(savedir, "wake_psd.pdf")
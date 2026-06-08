using JLD2
using Statistics
using Plots
using Thesis
using FFTW
using DSP

# Run directory holding U_hybrid_inline.jld2 / U_ref_inline.jld2
# savedir = "data/saved_models/inline_runs_hpc/base"
savedir = "data/inline_runs/2026-06-09_00-26"

hybrid_path = joinpath(savedir, "U_hybrid_inline.jld2")
ref_path    = joinpath(savedir, "U_ref_inline.jld2")



# --- wake probe location (EDIT ME) ---------------------------------------
# Cylinder (see circle_shedding_biot): center at (n/4, m/2), diameter D = m/8.
# We put the probe a few diameters downstream, slightly off the centerline,
# and read the cross-stream velocity (comp 2) which oscillates at the
# vortex-shedding frequency.
x_downstream_D = 2.0   # distance behind cylinder center, in diameters
y_offset_D     = 0.  # offset from the centerline, in diameters
component      = 1     # 1 = streamwise u, 2 = cross-stream v
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

p = plot(t_ref)
plot!(t_hyb)
display(p)
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

fs_ref = 1 / mean(diff(t_ref))           # sampling frequency from the time stamps
fs_hyb = 1 / mean(diff(t_hyb))

f_ref, P_ref = welch_psd(v_ref, fs_ref; nseg=n_segments, overlap=overlap)
f_hyb, P_hyb = welch_psd(v_hyb, fs_hyb; nseg=n_segments, overlap=overlap)

# --- plots ------------------------------------------------------------------
comp_label = component == 1 ? "u" : "v"

# shared publication styling: serif font, boxed frame, inward minor ticks
default(fontfamily = "Computer Modern", framestyle = :box, grid = false,
        tick_direction = :in, minorticks = true, widen = true,
        tickfontsize = 10, guidefontsize = 13, legendfontsize = 10,
        foreground_color_legend = :black, background_color_legend = :white)

ref_color = :black
hyb_color = RGB(0.85, 0.45, 0.0)

# (1) probe velocity signal
p_sig = plot(t_ref, v_ref; label = "reference", color = ref_color, lw = 1.6,
             xlabel = "\$t\$", ylabel = "\$$(comp_label)\$", legend = :topright,
             size = (560, 320))
plot!(p_sig, t_hyb, v_hyb; label = "hybrid", color = hyb_color, lw = 1.6, ls = :dash)
# plot!(p_sig, t_train, v_train)
savefig(p_sig, joinpath(savedir, "wake_signal.png"))

# (2) Welch power spectrum
p_psd = plot(f_ref[2:end], P_ref[2:end]; label = "reference", color = ref_color,
             lw = 1.6, xscale = :log10, yscale = :log10,
             xlabel = "frequency  \$f\$", ylabel = "PS(\$$(comp_label)\$)",
             legend = :bottomleft, size = (560, 460))
plot!(p_psd, f_hyb[2:end], P_hyb[2:end]; label = "hybrid", color = hyb_color,
      lw = 1.6, ls = :dash)
savefig(p_psd, joinpath(savedir, "wake_psd.png"))

display(p_sig)
display(p_psd)
@info "Saved" signal=joinpath(savedir, "wake_signal.png") psd=joinpath(savedir, "wake_psd.png")


using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using LaTeXStrings


root_path = is_hpc() ? "/scratch/mfbclaeys" : ""

AE_path_tl1 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE_path_tl1 = joinpath(root_path, AE_path_tl1)

normalizer = load_normalizer(AE_path_tl1)
ae_bundle, ae_args = load_trained_AE(AE_path_tl1)

node_path = "data/saved_models/NODE/16/RE2500/TL1_E500_curldiv_MS_Adam_250/node_params.jld2"
node_path = joinpath(root_path, node_path)
node, node_args = load_node(node_path)

z, t, tspan = Thesis.get_latent_vectors(ae_bundle, normalizer, ae_args; downsample=node_args.downsample)


dims = (1, 4, 8, 12)
n = length(dims)

# Compute a shared y-limit from the data, or per-panel — your call
ymax = maximum(abs, z) * 1.15  # shared, tight to data
# round to a clean tick value
ymax = ceil(ymax * 10) / 10  # e.g. 0.8, 0.9, 1.0


plt = plot(
    layout       = (2, 2),
    size         = (1000, 320),          # narrower, slightly less tall → better panel aspect
    legend       = false,
    link         = :x,
    framestyle   = :box,
    grid         = :y,                  # horizontal gridlines only
    gridalpha    = 0.20,
    gridlinewidth = 0.5,
    foreground_color_axis  = :black,
    foreground_color_text  = :black,
    left_margin   = 6Plots.mm,
    right_margin  = 3Plots.mm,
    top_margin    = 1Plots.mm,
    # bottom_margin = 4Plots.mm,
)

for (i, d) in enumerate(dims)
    is_bottom = i > 2
    plot!(plt[i], t, z[d, :];
          lw        = 1.2,
          color     = :midnightblue,
          xlims     = extrema(t),
          ylims     = (-ymax, ymax),
          ylabel    = L"z_{%$d}",       # LaTeXStrings interpolation — renders correctly
          xlabel     = is_bottom ? L"t^{*}" : "",
          xformatter = is_bottom ? :auto : _ -> "",
          bottom_margin = is_bottom ? 6Plots.mm : 1Plots.mm,
          xguide_position = :bottom,   # default, but explicit

          yguidefontrotation = 0,       # horizontal y-label, reads better in stacked panels
          widen     = false,
    )
end

display(plt)
savefig(plt, "figs/latent_trajectories.pdf")

# %%

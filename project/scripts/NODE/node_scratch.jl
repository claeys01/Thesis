using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using LaTeXStrings
using Printf

# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E1000-100div_100curl_ground_truth/checkpoint.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
normalizer = load_normalizer(AE_path)
ae_bundle, ae_args = load_trained_AE(AE_path)
ae_args.t_training = 16.603

# node_path = "data/saved_models/NODE/16/RE2500/E1000_100div100curl_ms_Adam_250_/node_params.jld2"
node_path = "data/NODE_models/NODE_May19-101631/node_params.jld2"
node, node_args = load_node(node_path)

# Encode the entire ground-truth trajectory (ae_t_train=Inf disables the t < t_training cutoff)
z_gt, t_gt, tspan_gt = Thesis.get_latent_vectors(
    ae_bundle, normalizer, ae_args;
    downsample=node_args.downsample, ae_t_train=Inf,
)

# Roll the NODE forward starting from t = t_start using the ground-truth latent at that time
t_start = 16.603
i_start = argmin(abs.(t_gt .- t_start))
t_roll  = t_gt[i_start:end]
z_pred  = Thesis.predict_array(node, z_gt[:, i_start]; t=t_roll)

dims = (1, 4, 8, 12)
ymax = maximum(abs, z_gt) * 1.15
ymax = ceil(ymax * 10) / 10

layout = @layout [grid(2, 2){0.92h}; b{0.08h}]
plt = plot(
    layout       = layout,
    size         = (1000, 360),
    legend       = false,
    framestyle   = :box,
    grid         = :y,
    gridalpha    = 0.20,
    gridlinewidth = 0.5,
    foreground_color_axis  = :black,
    foreground_color_text  = :black,
    left_margin   = 6Plots.mm,
    right_margin  = 3Plots.mm,
    top_margin    = 1Plots.mm,
    guidefontsize=10, tickfontsize=8, legendfontsize=9
)

for (i, d) in enumerate(dims)
    is_bottom = i > 2

    vspan!(plt[i], [t_gt[1], ae_args.t_training];
        fillcolor=:green, alpha=0.075, linealpha=0, label="")
    vspan!(plt[i], [ae_args.t_training, t_gt[end]];
        fillcolor=:purple, alpha=0.075, linealpha=0, label="")

    plot!(plt[i], t_gt, z_gt[d, :];
          lw        = 1.2,
          color     = :midnightblue,
          label     = "",
          xlims     = extrema(t_gt),
          ylims     = (-ymax, ymax),
          ylabel    = L"z_{%$d}",
          xlabel    = is_bottom ? L"t^{*}" : "",
          xformatter = is_bottom ? :auto : _ -> "",
          bottom_margin = is_bottom ? 6Plots.mm : 1Plots.mm,
          xguide_position = :bottom,
          yguidefontrotation = 0,
          widen     = false,
    )

    plot!(plt[i], t_roll, z_pred[d, :];
          lw        = 1.2,
          color     = :crimson,
          linestyle = :dash,
          label     = "",
    )
end

# Bottom legend strip: invisible axes, entries laid out in a single row
plot!(plt[5];
      framestyle = :none, grid = false, showaxis = false, ticks = nothing,
      xlims = (0, 1), ylims = (0, 1),
      legend = :top, legend_columns = -1, legendfontsize = 8,
      foreground_color_legend = nothing, background_color_legend = nothing,
      top_margin = 0Plots.mm, bottom_margin = 0Plots.mm)
plot!(plt[5], [NaN], [NaN]; color=:midnightblue, lw=1.2, label="ground truth")
plot!(plt[5], [NaN], [NaN]; color=:crimson, lw=1.2, linestyle=:dash, label="NODE rollout")
plot!(plt[5], [NaN], [NaN]; seriestype=:shape, fillcolor=:green, fillalpha=0.075, linealpha=0, label="train/val region")
plot!(plt[5], [NaN], [NaN]; seriestype=:shape, fillcolor=:purple, fillalpha=0.075, linealpha=0, label="test region")

display(plt)
savefig(plt, "figs/node_rollout_vs_gt.pdf")


# ── KNN OOD score over t* for varying k ──
train_mask = t_gt .< ae_args.t_training
k = 5
knn = fit_knn_ood(z_gt[:, train_mask]; k=k, q=0.99)

score_gt   = [KNN_score(knn, z_gt[:, i])   for i in axes(z_gt, 2)]
score_pred = [KNN_score(knn, z_pred[:, i]) for i in axes(z_pred, 2)]

cutoff_rel = findfirst(s -> s > knn.threshold, score_pred)
cutoff_t   = cutoff_rel === nothing ? nothing : t_roll[cutoff_rel]

knn_plt = plot(
    size         = (1000, 250),
    dpi          = 400,
    framestyle   = :box,
    grid         = :y,
    gridalpha    = 0.20,
    gridlinewidth = 0.5,
    foreground_color_axis  = :black,
    foreground_color_text  = :black,
    left_margin   = 6Plots.mm,
    right_margin  = 3Plots.mm,
    top_margin    = 2Plots.mm,
    bottom_margin = 6Plots.mm,
    legend        = :topleft,
    legendfontsize = 8,
    foreground_color_legend = nothing,
    background_color_legend = nothing,
    guidefontsize = 10, tickfontsize = 8,
    xlabel        = L"t^{*}",
    ylabel        = "KNN score",
    title         = "k = $k",
    titlefontsize = 10,
    xlims         = extrema(t_gt),
    widen         = false,
)
vspan!(knn_plt, [t_gt[1], ae_args.t_training];
        fillcolor=:green, alpha=0.075, linealpha=0, label="")
vspan!(knn_plt, [ae_args.t_training, t_gt[end]];
        fillcolor=:purple, alpha=0.075, linealpha=0, label="")
plot!(knn_plt, t_gt,   score_gt;   lw=1, color=:black,      label="GT KNN score")
plot!(knn_plt, t_roll, score_pred; lw=1, color=:dodgerblue, label="rollout KNN score")
hline!(knn_plt, [knn.threshold]; lw=1.5, ls=:dash, color=:firebrick,
        label=@sprintf("threshold (q=0.99) = %.3f", knn.threshold))
cutoff_t !== nothing && vline!(knn_plt, [cutoff_t]; lw=1.25, ls=:dash, color=:firebrick, label="")

display(knn_plt)
savefig(knn_plt, "figs/node_knn_score_k$(k).pdf")
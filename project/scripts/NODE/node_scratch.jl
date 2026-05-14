using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots


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
ylims = (-1, 1)
n = length(dims)
default(fontfamily="Computer Modern", titlefontsize=11,
        guidefontsize=10, tickfontsize=8, legendfontsize=9)

plt = plot(layout=(n, 1), size=(700, 750), legend=false,
           link=:x, framestyle=:box,
           grid=true, gridalpha=0.18, gridlinewidth=0.6,
           foreground_color_axis=:black, foreground_color_text=:black,
           left_margin=8Plots.mm, right_margin=6Plots.mm,
           top_margin=2Plots.mm, bottom_margin=2Plots.mm)

for (i, d) in enumerate(dims)
    plot!(plt[i], t, z[d, :];
          lw=1.4, color=:steelblue,
          xlims=(minimum(t), maximum(t)),
          ylims=ylims,
          ylabel="\$z_{$(d)}\$",
          yguidefontsize=14,
          xlabel=i == n ? "\$t^*\$" : "",
          xformatter=i == n ? :auto : _ -> "",
          widen=false)
    hline!(plt[i], [0]; lw=0.6, color=:gray60, ls=:dash, label="")
end

display(plt)

# savefig(plt, "latent_trajectories.pdf")

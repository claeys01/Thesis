using Thesis
using Thesis: get_NODE_data, load_node, predict_array, region_spans!
using Plots
using JLD2
using Printf

# params_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
params_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"

latent_idx     = [1, 10]                 # which latent components to show
total_downsample = -1                            # -1 = no downsampling
out_path       = "data/figures/node_rollout.png"

train_node, args = load_node(params_path; verbose=true)

z_total, t_total, tspan_total, z0_total =
    get_NODE_data(args.total_latent_path; downsample=total_downsample, verbose=true)
_, t_train, _, _ =
    get_NODE_data(args.train_latent_path; downsample=args.downsample, verbose=false)
_, t_test,  _, _ =
    get_NODE_data(args.test_latent_path;  downsample=args.test_downsample,  verbose=false)

rollout_node = deepcopy(train_node)
rollout_node.t     = t_total
rollout_node.tspan = tspan_total
ẑ_total = predict_array(rollout_node, z0_total)

mae = mean(abs.(z_total .- ẑ_total))
@info "Rollout finished" size(ẑ_total) MAE=mae

palette  = [:steelblue, :crimson, :seagreen, :darkorange, :purple, :goldenrod]
lw_truth = 2.4
lw_pred  = 1.7
α_truth  = 0.95
α_pred   = 0.9

n = length(latent_idx)
ncols = 1
nrows = length(latent_idx)
legend_bool = true
subplots = map(enumerate(latent_idx)) do (k, i)
    c = palette[(k - 1) % length(palette) + 1]
    sp = plot(
        framestyle = :box,
        grid       = true,
        ylims = (-0.75, 0.75),
        xlims=(0, 50),
        minorgrid  = true,
        gridalpha  = 0.25,
        guidefont  = font(11),
        tickfont   = font(9),
        titlefont  = font(12),
        legendfontsize = 8,
        legend     = :topright,
        foreground_color_legend = :black,
        background_color_legend = RGBA(1, 1, 1, 0.85),
        title      = "z$(i)",
        xlabel     = k > (nrows - 1) * ncols ? "time" : "",
        ylabel     = ((k - 1) % ncols == 0) ? "latent value" : "",
    )
    region_spans!(sp, t_train, t_test)
    plot!(sp, t_total, z_total[i, :];
        color = c, lw = lw_truth, alpha = α_truth, label = "truth")
    plot!(sp, t_total, ẑ_total[i, :];
        color = c, lw = lw_pred, alpha = α_pred, linestyle = :dash, label = "NODE")
    plot!(sp, legend= k == 1 ? :topright : nothing)
    # if legend_bool
    #     plot!(sp, legend=:topright)
    #     legend_bool = false
    # else
    #     plot!(sp, legend=nothing)
    # end

    sp
end

plt = plot(subplots...;
    layout        = (nrows, ncols),
    size          = (1200, 320 * nrows),
    # plot_title    = @sprintf("NODE rollout vs ground truth   (MAE = %.3g)", mae),
    # plot_titlefont = font(14),
    left_margin   = 6Plots.mm,
    bottom_margin = 5Plots.mm,
    top_margin    = 4Plots.mm,
    dpi=350,
)

display(plt)
mkpath(dirname(out_path))
savefig(plt, out_path)
@info "Saved figure" out_path

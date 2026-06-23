using Plots, StatsPlots, LaTeXStrings, Printf
# using Thesis
# ---- data: per group size (ordered 10, 15, 20) -----------------------------
group_size = [10, 15, 20]
avg_rollout_length = [359, 408, 681]

rel_err_cd_reg2 = [6.69521, 0.1694068863, 15.1021]
rel_err_cl_reg2 = [13.1882, 11.36139136, 4.58584]


rst_uu =   [0.011,  0.00641784, 0.008]
rst_vv =   [0.028,  0.0138134,  0.018]
rst_uv =   [0.005,  0.00406987, 0.005]
rst_r_uu = [0.953,  0.978895,   0.9603]
rst_r_vv = [0.936,  0.979051,   0.9777]
rst_r_uv = [0.9075, 0.927507,   0.8969]


plot_kwargs = (
    xlabel = L" n_s",
    ylabel = L"$\varepsilon$",
    xticks = (group_size, string.(group_size)),
    # xscale = :log2,
    legend = :topright,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    minorgrid = false,
    legendfontsize = 11,
    guidefontsize = 16,
    tickfontsize = 10,
    markersize = 6,
    left_margin = 3Plots.mm,
    bottom_margin = 4Plots.mm,
    linewidth = 2,
    size = (350, 200),
    dpi = 500,
)

# correlation-panel styling (shares the structural kwargs, swaps the y label)
corr_kwargs = merge(plot_kwargs, (ylabel = L"$\rho$",))

# ---- mean-flow comparison --------------------------------------------------
rel_err = groupedbar(string.(group_size), [rel_err_cd_reg2 rel_err_cl_reg2];
    color = [:royalblue :seagreen], bar_width = 0.7, lw = 0,
    label = [L"\langle C_D \rangle" L"C_L^{\mathrm{rms}}"], plot_kwargs...,
    xticks = :auto, ylabel = L"$e_{\mathrm{rel}}$",
)

rel_err_comp = plot(rel_err; layout = (1, 1), size = (600, 400), dpi = 300)

display(rel_err_comp)

rst_eps = plot(group_size, [rst_uu rst_vv rst_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    plot_kwargs...)
rst_r = plot(group_size, [rst_r_uu rst_r_vv rst_r_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    ylims = (0.84, 1.0), yticks = 0.84:0.04:1.0,
    corr_kwargs..., 
    legend=:bottomright,
)
rst_comp = plot(rst_eps, rst_r; layout = (1, 2), size = (1000, 400), dpi = 300)
display(rst_comp)
savefig(rst_comp, "project/scripts/Thesis_stats_plots/gs_comp.pdf")
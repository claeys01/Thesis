using Plots, LaTeXStrings, Printf

# ---- data: MAE per latent dimension ----------------------------------------
Nz = [8, 16, 32]

meanflow_u = [0.0109, 0.0094, 0.0102]      # ⟨u⟩
meanflow_v = [0.0078, 0.0053, 0.0074]      # ⟨v⟩

rst_uu = [0.0100, 0.0064, 0.0049]          # ⟨u'u'⟩
rst_vv = [0.0274, 0.0138, 0.0103]          # ⟨v'v'⟩
rst_uv = [0.0065, 0.0041, 0.0034]          # ⟨u'v'⟩



plot_kwargs = (
    xlabel = L" $N_z$",
    ylabel = "MAE",
    xticks = (Nz, string.(Nz)),
    xscale = :log2,
    legend = :topright,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    minorgrid = false,
    legendfontsize = 11,
    guidefontsize = 12,
    tickfontsize = 10,
    markersize = 6,
    linewidth = 2,
    size = (640, 460),
    dpi = 300,
)

# ---- mean-flow comparison --------------------------------------------------
meanflow_comp = plot(Nz, [meanflow_u meanflow_v];
    color = [:royalblue :seagreen], ls = :dash, marker = [:circle :diamond],
    label = [L"\langle u\rangle" L"\langle v\rangle"], plot_kwargs...)
display(meanflow_comp)

# ---- Reynolds-stress comparison --------------------------------------------
rst_comp = plot(Nz, [rst_uu rst_vv rst_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    plot_kwargs...)
display(rst_comp)

# ---- bar charts ------------------------------------------------------------
using StatsPlots

bar_kwargs = (
    xlabel = L"$N_z$",
    ylabel = "MAE",
    legend = :topright,
    framestyle = :box,
    grid = true,
    gridalpha = 0.3,
    minorgrid = false,
    legendfontsize = 11,
    guidefontsize = 12,
    tickfontsize = 10,
    linewidth = 0.5,
    bar_width = 0.7,
    size = (640, 460),
    dpi = 300,
)

nz_ticks = string.(Nz)

meanflow_bar = groupedbar(nz_ticks, [meanflow_u meanflow_v];
    color = [:royalblue :seagreen],
    label = [L"\langle u\rangle" L"\langle v\rangle"],
    bar_kwargs...)
display(meanflow_bar)

rst_bar = groupedbar(nz_ticks, [rst_uu rst_vv rst_uv];
    color = [:royalblue :seagreen :firebrick],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    bar_kwargs...)
display(rst_bar)
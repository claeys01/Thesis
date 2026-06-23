using Plots, LaTeXStrings, Printf

# ---- data: MAE per latent dimension ----------------------------------------
Nz = [8, 16, 32]

meanflow_u = [0.0109, 0.0094, 0.0102]      # ⟨u⟩
meanflow_v = [0.0078, 0.0053, 0.0074]      # ⟨v⟩

rst_uu = [0.0100, 0.0064, 0.0049]          # ⟨u'u'⟩
rst_vv = [0.0274, 0.0138, 0.0103]          # ⟨v'v'⟩
rst_uv = [0.0065, 0.0041, 0.0034]          # ⟨u'v'⟩

# ---- correlation coefficient r per latent dimension ------------------------
meanflow_r_u = [0.9914, 0.9932, 0.9945]    # ⟨u⟩
meanflow_r_v = [0.9861, 0.9928, 0.9914]    # ⟨v⟩

rst_r_uu = [0.9595, 0.9789, 0.9913]        # ⟨u'u'⟩
rst_r_vv = [0.9426, 0.9791, 0.9911]        # ⟨v'v'⟩
rst_r_uv = [0.8672, 0.9275, 0.9525]        # ⟨u'v'⟩

S_loop = [2.51, 1.48, 1.64]
S_eff = [0.113, 0.058, 0.069]

plot_kwargs = (
    xlabel = L" $N_z$",
    ylabel = L"$\varepsilon$",
    xticks = (Nz, string.(Nz)),
    xscale = :log2,
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
    linewidth = 1,
    size = (350, 200),
    dpi = 300,
)

# correlation-panel styling (shares the structural kwargs, swaps the y label)
corr_kwargs = merge(plot_kwargs, (ylabel = L"$\rho$",))

# ---- mean-flow comparison --------------------------------------------------
mf_eps = plot(Nz, [meanflow_u meanflow_v];
    color = [:royalblue :seagreen], ls = :dash, marker = [:circle :diamond],
    label = [L"\langle u\rangle" L"\langle v\rangle"], plot_kwargs...)
mf_r = plot(Nz, [meanflow_r_u meanflow_r_v];
    color = [:royalblue :seagreen], ls = :dash, marker = [:circle :diamond],
    label = [L"\langle u\rangle" L"\langle v\rangle"],
    ylims = (0.95, 1.0), yticks = 0.95:0.01:1.0, corr_kwargs...)
meanflow_comp = plot(mf_eps, mf_r; layout = (1, 2), size = (1000, 400), dpi = 300)
display(meanflow_comp)

# ---- Reynolds-stress comparison --------------------------------------------
rst_eps = plot(Nz, [rst_uu rst_vv rst_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    plot_kwargs...)
rst_r = plot(Nz, [rst_r_uu rst_r_vv rst_r_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    ylims = (0.84, 1.0), yticks = 0.84:0.04:1.0,
    corr_kwargs..., 
    legend=:bottomright,
)
rst_comp = plot(rst_eps, rst_r; layout = (1, 2), size = (1000, 400), dpi = 300)
display(rst_comp)



using Plots, LaTeXStrings, Printf
using Thesis
# ---- data: MAE per latent dimension ----------------------------------------
epochs = [250, 500, 750]

meanflow_u = [0.0098, 0.0094, 0.0091]      # ⟨u⟩
meanflow_v = [0.0061, 0.0053, 0.0055]      # ⟨v⟩

rst_uu = [0.0080, 0.0064, 0.0055]          # ⟨u'u'⟩
rst_vv = [0.0193, 0.0138, 0.0116]          # ⟨v'v'⟩
rst_uv = [0.0038, 0.0041, 0.0034]          # ⟨u'v'⟩

# ---- correlation coefficient r per latent dimension ------------------------
meanflow_r_u = [0.9926, 0.9932, 0.9936]    # ⟨u⟩
meanflow_r_v = [0.9893, 0.9928, 0.9924]    # ⟨v⟩

rst_r_uu = [0.9678, 0.9789, 0.9860]        # ⟨u'u'⟩
rst_r_vv = [0.9632, 0.9791, 0.9881]        # ⟨v'v'⟩
rst_r_uv = [0.9347, 0.9275, 0.9491]        # ⟨u'v'⟩

S_loop = [2.51, 1.48, 1.64]
S_eff = [0.113, 0.058, 0.069]

plot_kwargs = (
    xlabel = L" Epochs",
    ylabel = L"$\varepsilon$",
    xticks = (epochs, string.(epochs)),
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
    dpi = 300,
)

# correlation-panel styling (shares the structural kwargs, swaps the y label)
corr_kwargs = merge(plot_kwargs, (ylabel = L"$\rho$",))

# ---- mean-flow comparison --------------------------------------------------
mf_eps = plot(epochs, [meanflow_u meanflow_v];
    color = [:royalblue :seagreen], ls = :dash, marker = [:circle :diamond],
    label = [L"\langle u\rangle" L"\langle v\rangle"], plot_kwargs...)
mf_r = plot(epochs, [meanflow_r_u meanflow_r_v];
    color = [:royalblue :seagreen], ls = :dash, marker = [:circle :diamond],
    label = [L"\langle u\rangle" L"\langle v\rangle"],
    ylims = (0.95, 1.0), yticks = 0.95:0.01:1.0, corr_kwargs...)
meanflow_comp = plot(mf_eps, mf_r; layout = (1, 2), size = (1000, 400), dpi = 300)
display(meanflow_comp)

# ---- Reynolds-stress comparison --------------------------------------------
rst_eps = plot(epochs, [rst_uu rst_vv rst_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    plot_kwargs...)
rst_r = plot(epochs, [rst_r_uu rst_r_vv rst_r_uv];
    color = [:royalblue :seagreen :firebrick], ls = :dash,
    marker = [:circle :diamond :utriangle],
    label = [L"\langle u'u'\rangle" L"\langle v'v'\rangle" L"\langle u'v'\rangle"],
    ylims = (0.84, 1.0), yticks = 0.84:0.04:1.0,
    corr_kwargs..., 
    legend=:bottomright,
)
rst_comp = plot(rst_eps, rst_r; layout = (1, 2), size = (1000, 400), dpi = 300)
display(rst_comp)
savefig(rst_comp, joinpath("data/saved_models/inline_runs_hpc/inline_sweep_final", "rst_comp.pdf") )

AE250_checkpoint = "data/saved_models/inline_runs_hpc/inline_sweep_final/aeEpochs_250/AE_Jun05-1623__E250_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE500_checkpoint = "data/saved_models/inline_runs_hpc/base/AE_Jun05-1553__E500_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
AE750_checkpoint = "data/saved_models/inline_runs_hpc/inline_sweep_final/aeEpochs_750/AE_Jun05-1644__E750_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
checkpoints = [AE250_checkpoint, AE500_checkpoint, AE750_checkpoint]

train_data = "data/saved_models/inline_runs_hpc/base/U_inline.jld2"
z_arr = []
t_arr = []
for checkpoint in checkpoints
    ae_bundle, ae_args = load_trained_AE(checkpoint)
    normalizer = load_normalizer(checkpoint)
    z, t, tspan, z0 = Thesis.get_latent_vectors(ae_bundle, normalizer, ae_args; full_data_path = train_data)
    push!(z_arr, z)
    push!(t_arr, t)
    GC.gc()
end

@show size(z_arr[1])
idx_samples = [1, 4, 8, 12]
zlo = minimum(minimum(view(z, idx_samples, :)) for z in z_arr)
zhi = maximum(maximum(view(z, idx_samples, :)) for z in z_arr)

latent_kwargs =(
    # ylims=(zlo, zhi),
    dpi=400,
    size=(750,250),
    xlabel=L"t^*",
    # ylabel=L"\mathbf{z}",
    legendfontsize = 8,
    guidefontsize = 20,
    framestyle   = :box,
    gridalpha    = 0.20,
    gridlinewidth = 0.5,
    foreground_color_axis  = :black,
    foreground_color_text  = :black,
    left_margin   = 8Plots.mm,
    right_margin  = 3Plots.mm,
    top_margin    = 1Plots.mm,
    bottom_margin = 9Plots.mm,
    xguide_position = :bottom,   # default, but explicit
)

colors = [:red, :blue, :black]
linestyles = [:dash, :dot, :dashdot, :dashdotdot]
ae_labels = ["250 epochs", "500 epochs", "750 epochs"]

plots_arr = []
for (j, idx) in enumerate(idx_samples)
    p = plot(; ylabel = L"z_{%$idx}", latent_kwargs...)
    for (i, z) in enumerate(z_arr)
        plot!(p, t_arr[i], z[idx, :]; color = colors[i], ls =:solid,
              label = j == 1 ? ae_labels[i] : "", lw=1.25)
        if idx < 7
            plot!(p, xlabel="", xformatter=_ -> "", bottom_margin = -2Plots.mm)
        end
    end
    push!(plots_arr, p)
end

latent_comp = plot(plots_arr...; layout = (2, 2), size = (1500, 500), dpi = 400)
display(latent_comp)
savefig(latent_comp, joinpath("data/saved_models/inline_runs_hpc/inline_sweep_final", "epoch_latent_comp.pdf"))
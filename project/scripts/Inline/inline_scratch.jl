using Thesis
using JLD2
using Plots

U_inline_path = "data/inline_runs/2026-05-28_19-40/U_inline.jld2"
path = "data/saved_models/inline_runs_hpc/ae500_lat16_nit250_gs10/hybrid_state.jld2"


@load path res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL2_E300_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"

# AE_path = "data/saved_models/inline_runs_hpc/ae500_lat16_nit250_gs10/AE_May26-0337__E100_HW256x256_C4to2_nc6_nd1_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
ae_bundle, ae_args = Thesis.load_trained_AE(AE_path)
# @load hybrid_state_path AE_path

ae_bundle, ae_args = Thesis.load_trained_AE(AE_path)
simdata = load_simdata(U_inline_path)
ae_args.train_downsample=500

# plt, _ = Thesis.velocity_flood(sim; colorbar=true)
# display(plt)

# Thesis.preprocess_data!(simdata)

ae_args.t_training = 0.90 *simdata.time[end]
idxs = Thesis.get_idxs(simdata, ae_args)
plt = Thesis.train_force_plot(simdata;
    train_idx = idxs.train_idx,
    val_idx   = idxs.val_idx,
    test_idx  = idxs.test_idx,
    show_zeros = false,
)
simdata_full = load_simdata(ae_args.full_data_path)
full_time = simdata_full.time
full_drag, full_lift = first.(simdata_full.force), last.(simdata_full.force)


idx_end = findall(t -> t ≥ simdata.time[end-1], full_time)

# plot!(plt, full_time[idx_end], full_drag[idx_end], color=:red)
# plot!(plt, full_time[idx_end], full_lift[idx_end], color=:blue)

plot!(plt, xlim=(0, simdata.time[end]), legend_column=2, legend=:bottomleft, legendfontsize=6)
display(plt)

GC.gc()
simdata_full = nothing
savepath = joinpath(dirname(AE_path), "training_force_replot.pdf")
savefig(plt, savepath)
@info "training force plot saved" savepath



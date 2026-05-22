using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using Printf


# @load "data/inline_runs/2026-05-22_11-13/accel_results.jld2" res
@load "data/inline_runs/2026-05-22_11-13/hybrid_state.jld2" res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir

plot_forces_comparison(res, params.t_accel_end)
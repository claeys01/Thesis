using Thesis
using JLD2
using Statistics

path = "data/saved_models/inline_runs_hpc/temp/hybrid_state.jld2"

@load path res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir

print_metrics(res; pred_label="(flexible OOD)",
    avg_steps_per_pred=isempty(n_integrs) ? nothing : mean(n_integrs),
    sim_meanflow=sim_meanflow, ref_meanflow=ref_meanflow)
# display(plot_rst_comparison(sim_meanflow, ref_meanflow))
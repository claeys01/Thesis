using Thesis
using Plots, LaTeXStrings

node_path = "data/saved_models/inline_runs_hpc/base/NODE_Jun05-160822/node_params.jld2"
AE_path = "data/saved_models/inline_runs_hpc/base/AE_Jun05-1553__E500_HW256x256_C4to2_nc5_nd1_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"

ae_bundle, ae_args = load_trained_AE(AE_path)
normalizer = load_normalizer(AE_path)
node, node_args = load_node(node_path)

full_path = "data/datasets/RE2500/2e8/U_128_full.jld2"
base_path = "data/saved_models/inline_runs_hpc/base/U_inline.jld2"
z, t, tspan, z0 = Thesis.get_latent_vectors(ae_bundle, normalizer, ae_args; full_data_path=base_path)
z_ext, t_ext, _, _ = Thesis.get_latent_vectors(ae_bundle, normalizer, ae_args; full_data_path=full_path, ae_t_train=20)


knn = fit_knn_ood(z)

# ----------------------------------------------------------------------------
# Rollout assessment plot
# ----------------------------------------------------------------------------
# Two single-shooting rollouts of the trained NODE, each in its own column:
#   1. started from t* = 0          (covers the whole run, includes training region)
#   2. started from t* = t_train    (pure extrapolation up to t_run)
# Ground truth (z) is always drawn over the full run up to t_run.
# Underneath each rollout we plot the KNN OOD score of the predicted states.

# ---- editable parameters ----
t_train = 17.5      # NODE training horizon
t_run   = 20.0      # full run / ground-truth horizon (data already extends to here)
n_show  = 4         # number of latent components to overlay

palette = [:black, :red, :blue, :green, :purple, :orange, :yellow]
latent_dim  = size(z, 1)
idx_samples = round.(Int, range(1, stop=latent_dim, length=n_show))   # which latents to draw

# rollout start indices in the time vector t
start_idx_0  = 1
start_idx_tr = argmin(abs.(t .- t_train))

# Integrate the NODE from a given start index to the end of the run.
# Returns the time vector of the rollout and the (latent_dim × n_t) prediction.
function rollout(node, z, t, start_idx)
    t_chunk = t[start_idx:end]
    z0_loc  = z[:, start_idx]
    pred    = Array(predict_array(node, z0_loc; t=t_chunk))
    return t_chunk, pred
end

# KNN OOD score for every predicted latent state along a rollout.
knn_curve(knn, pred) = [KNN_score(knn, pred[:, j]) for j in axes(pred, 2)]

# A single latent-trajectory panel: full ground truth (solid) + rollout prediction (dashed).
function plot_latent_rollout(z, t, t_chunk, pred; title="")
    p = plot(dpi=400, xlabel=L"t^*", ylabel=L"\mathbf{z}", title=title,
             legendfontsize=7, framestyle=:box, gridalpha=0.20, gridlinewidth=0.5,
             foreground_color_axis=:black, foreground_color_text=:black)
    for (s, li) in enumerate(idx_samples)
        c = palette[(s - 1) % length(palette) + 1]
        plot!(p, t,       z[li, :];     color=c, linestyle=:solid, label = s == 1 ? "z (truth)" : "")
        plot!(p, t_chunk, pred[li, :];  color=c, linestyle=:dash,  label = s == 1 ? "z̃ (pred)" : "")
    end
    return p
end

# A single KNN-score panel aligned under its trajectory panel.
function plot_knn_rollout(t_chunk, score; threshold=nothing, title="")
    p = plot(dpi=400, xlabel=L"t^*", ylabel="KNN score", title=title,
             legendfontsize=7, framestyle=:box, gridalpha=0.20, gridlinewidth=0.5,
             foreground_color_axis=:black, foreground_color_text=:black)
    plot!(p, t_chunk, score; color=:black, label="KNN score")
    threshold === nothing || hline!(p, [threshold]; color=:red, linestyle=:dash, label="threshold")
    return p
end

# ---- build the two rollouts ----
t0,  pred0  = rollout(node, z, t, start_idx_0)    # from t* = 0
ttr, predtr = rollout(node, z_ext, t_ext, start_idx_tr)   # from t* = t_train

score0  = knn_curve(knn, pred0)
scoretr = knn_curve(knn, predtr)

# ---- assemble 2×2 figure: columns = rollouts, rows = (trajectory, KNN score) ----
p_traj0  = plot_latent_rollout(z, t, t0,  pred0;  title="Rollout from t* = 0")
p_knn0   = plot_knn_rollout(t0,  score0;  threshold=knn.threshold)
plot!(p_knn0, t, knn_curve(knn, z))

println(0.326/0.999 - maximum(score0))

fig = plot(p_traj0, p_knn0;
           layout=grid(2, 1, heights=[0.6, 0.4]),
           size=(1200, 650),
           left_margin=6Plots.mm, right_margin=3Plots.mm,
           top_margin=2Plots.mm, bottom_margin=6Plots.mm)

out_path = "data/saved_models/inline_runs_hpc/base/NODE_Jun05-160822/rollout_per_trajectory.pdf"
savefig(fig, out_path)
@info "Saved rollout assessment plot" out_path
display(fig)

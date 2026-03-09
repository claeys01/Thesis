using Thesis
using Plots
using LinearAlgebra
using Statistics
using KernelDensity

node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

latent_data = Thesis.load_datasets(aenode.node_args)

simdata = load_simdata(aenode.ae_args.full_data_path)



t₀ = simdata.time[1]

rollout_range = collect(1:3000)
time = simdata.time[rollout_range]
z0 = latent_data.z_total[:, rollout_range[1]]
sol = Thesis.predict_array(aenode.NODE,  z0; t=time, onlysol=false)

ẑ = Array(sol[1])

ẑ_norms = norm.(eachcol(ẑ))

plt = plot()
for i in 1:16
    plot!(plt, time, ẑ[i, :])
end
plot!(plt, time, mean(ẑ, dims=1)[1, :], lw=2, color=:black)
display(plt)

# Compute KDE of training latent trajectories
kde_model = kde(latent_data.z_total)  # KDE in latent space

# Evaluate log-likelihood of rollout trajectory under the KDE
logpdf_rollout = logpdf(kde_model, ẑ)  # shape: (16, 3000) or similar

# Compute statistics
mean_logpdf = mean(logpdf_rollout)
min_logpdf = minimum(logpdf_rollout)
std_logpdf = std(logpdf_rollout)

println("KDE Log-Likelihood Statistics:")
println("  Mean: $(round(mean_logpdf, digits=4))")
println("  Min:  $(round(min_logpdf, digits=4))")
println("  Std:  $(round(std_logpdf, digits=4))")

# Plot log-likelihood over time
plt_kde = plot(
    time, vec(mean(logpdf_rollout, dims=1));
    label="Mean log-likelihood",
    xlabel="tU/L",
    ylabel="log p(z)",
    title="KDE Log-Likelihood of Rollout Trajectory",
    framestyle=:box,
    linewidth=2,
    size=(700, 400)
)
hline!(plt_kde, [mean_logpdf]; label="Overall mean", linestyle=:dash, color=:red)
display(plt_kde)

# Count how many points fall below a threshold (e.g., mean - 2*std)
threshold = mean_logpdf - 2*std_logpdf
out_of_dist = count(<(threshold), logpdf_rollout)
pct_ood = 100 * out_of_dist / length(logpdf_rollout)

println("\nOut-of-Distribution Analysis:")
println("  Threshold (mean - 2σ): $(round(threshold, digits=4))")
println("  Points below threshold: $out_of_dist / $(length(logpdf_rollout)) ($(round(pct_ood, digits=2))%)")

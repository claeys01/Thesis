using Thesis


train()
# gc.gc()
# using Thesis
# using JLD2

# # Load the old data (as ReconstructedMutable)
# old_data = JLD2.load("data/datasets/RE2500/2e8/U_128_full.jld2", "simdata")

# # Convert to proper SimData
# simdata = SimData(
#     time = old_data.time,
#     Δt = old_data.Δt,
#     u = old_data.u,
#     μ₀ = old_data.μ₀,
#     force = old_data.force,
#     ε = old_data.ε,
#     period_ranges = old_data.period_ranges,
#     reordered_ranges = old_data.reordered_ranges,
#     single_period_idx = old_data.single_period_idx
# )

# # Re-save with correct type
# JLD2.save("data/datasets/RE2500/2e8/U_128_full.jld2", "simdata", simdata)
using JLD2
using Revise
using DiffEqFlux: group_ranges

includet("../AE/Lux_AE.jl")
includet("NODE_core.jl")

# simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")
# u = simdata.u[:, :, :, 1:500]
# u_batch = simdata.u[:, :, :, 200:300]
# # u = randn(256, 256, 2, 1000)
# simdata = nothing

# _, normalizer = normalize_batch(u; normalizer=nothing)
# u_norm, _ = normalize_batch(u_batch; normalizer=normalizer)

# @show vec(mean(u_norm, dims=(1,2,4)))   # should be ~ Float32[0, 0]
# @show vec(std(u_norm, dims=(1,2,4)))   # should be ~ Float32[0, 0]

# GC.gc()
args = NodeArgs()
z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
@show size(z)
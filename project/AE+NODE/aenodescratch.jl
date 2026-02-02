using Revise
using Random
using Plots
Random.seed!(42)
includet("AENODE.jl")

using TimerOutputs
to = TimerOutput()


sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=true)

@show sim.flow.Δt[1:end-1], sum(sim.flow.Δt[1:end-1])/sim.L

# sim_step!(sim)
# sim_info(sim)
# @show sim.flow.Δt[1:end-1], sum(sim.flow.Δt[1:end-1])/sim.L


node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

aenode = AENODE(AE_path, node_path)

#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)
# preprocess_data!(simdata)

# random_int = rand(1:1000)
random_int = 10

u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]
@show simdata.time[random_int:random_int+10]
@show simdata.Δt[random_int:random_int+10]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
# @show sim.flow.Δt, size(sim.flow.Δt)



plt1 = flood(sim.flow.u[:, :, 1];
border=:none, colorbar=true, framestyle=:none,
axis=nothing, ticks=false,
aspect_ratio=:equal, titlefontsize=8)

sim_step!(sim)
sim_info(sim)
@show sim.flow.Δt[1:end-1], sum(sim.flow.Δt[1:end-1])/sim.L

plt2 = flood(sim.flow.u[:, :, 1];
border=:none, colorbar=true, framestyle=:none,
axis=nothing, ticks=false,
aspect_ratio=:equal, titlefontsize=8)

plt3 = flood(simdata.u[:, :, 1, random_int+1];
border=:none, colorbar=true, framestyle=:none,
axis=nothing, ticks=false,
aspect_ratio=:equal, titlefontsize=8)

total_plt = plot(plt1, plt2, plt3; layout=(1,3))
display(total_plt)
simdata = nothing


@info "predicting 10 timesteps with aenode"
sim = predict_n(aenode, sim, 10; Δt=mean(sim.flow.Δt[10:end]))
sim_info(sim)
@show sim.flow.Δt[1:end-1], sum(sim.flow.Δt[1:end-1])/sim.L


sim_step!(sim)
sim_info(sim)
@show sim.flow.Δt[1:end-1], sum(sim.flow.Δt[1:end-1])/sim.L

plt = flood(sim.flow.u[:, :, 1];
border=:none, colorbar=true, framestyle=:none,
axis=nothing, ticks=false,
aspect_ratio=:equal, titlefontsize=8)
# display(plt)
# @info "predicting trajectory"
# @timeit to "predicting trajectory" û_traj = predict_n(aenode, u, μ₀, 10, t₀; return_traj=true)
# # @show size(û_traj)

# @info "predicting end of trajectory"
# @timeit to "predicting end of trajectory" û_end = predict_n(aenode, u, μ₀, 10, t₀; return_traj=false)

# @assert û_end == û_traj[:, :, :, end]

# show(to)
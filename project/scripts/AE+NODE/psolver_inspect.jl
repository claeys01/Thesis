using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs
using BiotSavartBCs

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
reset_timer!(to::TimerOutput)

# load aenode struct with trained neural ai models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)

function plot_pressure(sim::BiotSimulation)
    p = sim.flow.p
    px = WaterLily.flood(p[:, :], border=:none, colorbar=false, framestyle=:none, axis=nothing, ticks=false)
    # py = WaterLily.flood(p[:, :, 2], border=:none, colorbar=false, framestyle=:none, axis=nothing, ticks=false)
    # plt = plot(px, py, layout=(1, 2))
    px
end

# getting simulation with know initial condition
random_int = 1
u₀, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]
sim.flow.u .= u₀
append!(sim.flow.Δt, simdata.Δt[1:random_int-1])
sim_step!(sim; verbose=true)

u_first = copy(sim.flow.u)
p_first = copy(sim.flow.p)
# @show mean(p_first)
# sim.flow.p .= 0f0
# WaterLily.measure!(sim)

# to test pressure method, we set pressure field to zero, and run it. Velocity field should not change
p_pre = plot_pressurze(sim)

Thesis.impose_biot_bc!(sim)

# WaterLily.sim_step!(sim)
# temp = deepcopy(sim)
# WaterLily.sim_step!(temp)
# p_end = temp.flow.p
println("mean velocity before: $(mean(u_first)), and after $(mean(sim.flow.u)), rMAE: $(mean(abs, u_first .- sim.flow.u))")
println("mean pressure before: $(mean(p_first)), and after $(mean(sim.flow.p)), rMAE: $(mean(abs, p_first .- sim.flow.p))")

# predicting the same velocity field as in sim
# @show t₀, sim_time(sim), sim.flow.Δt
u_pred = predict_n(aenode, u₀, μ₀, 1, t₀; Δt=sim.flow.Δt[end])
# println("\nMAE between sim and pred: $(mean(abs, u_pred .- sim.flow.u[2:end-1, 2:end-1, :]))\n")

Thesis.insert_prediction!(sim, u_pred)
WaterLily.measure!(sim)
sim.flow.p .= 0f0

# # temp2 = deepcopy(sim)
# # WaterLily.sim_step!(temp2)
# # temp2.flow.p .= 0f0

# # WaterLily.sim_step!(temp2)

# # p_end_pred = temp2.flow.p
# p_pre = plot_pressure(sim)

# Thesis.impose_biot_bc!(sim)
WaterLily.sim_step!(sim)
WaterLily.sim_step!(sim)

# Thesis.impose_biot_bc!(sim)
# Thesis.impose_biot_bc!(sim)


p_aft = plot_pressure(sim)
plt = plot(p_pre, p_aft, layout=(1,2), colorbar=true, levels=20)
display(plt)


println("\noriginal mean velocity: $(mean(u_first)), and predicted $(mean(sim.flow.u)), rMAE: $(mean(abs, u_first .- sim.flow.u))")
println("original mean pressure: $(mean(p_first)), and predicted $(mean(sim.flow.p)), MAE: $(mean(abs, p_first .- sim.flow.p))")

println("\n#################################################################################################\n")

# simdata = nothing
# sim = nothing
# GC.gc()
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

random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]
sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int-1])
sim_step!(sim; verbose=true)
u_first = copy(sim.flow.u)
p_first = copy(sim.flow.p)


@show mean(u_first)
@show mean(p_first)

step = 0
sim_ref = deepcopy(sim)

function total_time(sim::BiotSimulation)
    sum(sim.flow.Δt)*sim.U/sim.L
end

pred_sim_time = total_time(sim) + (16 * 0.35)*sim.U/sim.L
while total_time(sim_ref) < pred_sim_time
    sim_step!(sim_ref)
    step +=1
end

predict_n!(sim, aenode, 16; Δt=0.35f0, impose_biot=false)
@show mean(sim.flow.u)
Thesis.impose_biot_bc!(sim)

# @show mean(u_first), mean(sim.flow.u)
# @show mean(p_first), mean(sim.flow.p)

@show mean(sim_ref.flow.u), mean(sim.flow.u)
@show mean(sim_ref.flow.p), mean(sim.flow.p)
@show total_time(sim_ref), total_time(sim)

# @show size(sim_ref.flow.Δt), size(sim.flow.Δt)
simdata = nothing
sim = nothing

GC.gc()


# @show get_forces(sim)

# predict_n!(sim, aenode, 16; Δt=0.35f0, impose_biot=true)

# @show get_forces(sim)

# WaterLily.logger("data/test_psolver")

# simstep=5

# function get_forces(sim::BiotSimulation)
#     raw_force = WaterLily.pressure_force(sim)
#     scaled_force = Float32.(raw_force./(0.5sim.L*sim.U^2)) # scale the forces!
#     return scaled_force
# end

# for _ in 1:simstep
#     sim_step!(sim; verbose=true)
#     forces = get_forces(sim)
#     @show forces
# end
# plt = WaterLily.plot_logger("data/test_psolver")
# display(plt)
# savefig("figs/psolver.png")
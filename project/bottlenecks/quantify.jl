using WaterLily
using Plots
using TimerOutputs
using Revise


to = TimerOutput()

includet("../vortex_shedding.jl")


sim_shedding = circle_shedding(mem=Array)
t_end = 50.0

@timeit to "sim_step!" begin
    gif = sim_gif!(sim_shedding;duration=t_end,clims=(-5,5),plotbody=true)
    # sim_step!(sim_shedding, t_end; verbose=true, timer=to)
end


# u = sim_shedding.flow.u[:,:,1] # x velocity
# ω = zeros(size(u));

# @inside ω[I] = WaterLily.curl(3,I,sim_shedding.flow.u)*sim_shedding.L/sim_shedding.U

# println(sum(abs, ω))
# println(sim_shedding.U)

# plt = flood(ω, clims=(-5,5), border=:none)
# display(plt)

# function get_forces!(sim,t)
#     sim_step!(sim,t,remeasure=false; verbose=true)
#     force = WaterLily.pressure_force(sim)
#     force./(0.5sim.L*sim.U^2) # scale the forces!
# end

# # Simulate through the time range and get forces
# time = 1:0.1:100 # time scale is sim.L/sim.U
# forces = [get_forces!(sim_shedding,t) for t in time];

# #Plot it
# plot(time,[first.(forces) last.(forces)],
#     labels=["drag" "lift"],
#     xlabel="tU/L",
#     ylabel="Pressure force coefficients")

# show(to)
display(gif)
# reset_timer!(to)
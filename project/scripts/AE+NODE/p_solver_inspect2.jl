using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs
using BiotSavartBCs
using Printf

import WaterLily: Vcycle!,smooth!, scale_u!, conv_diff!, udf!, accelerate!, BDIM!
import BiotSavartBCs: apply_grad_p!, biotBC!, fix_resid!, biotBC_r!, pflowBC!, BCTuple

sim = circle_shedding_biot(; mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

# Load data
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)
simdata = load_simdata(aenode.ae_args.full_data_path)

# Load snapshot at index 1
random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]


function get_forces(sim::BiotSimulation)
    raw_force = WaterLily.pressure_force(sim)
    scaled_force = Float32.(raw_force ./ (0.5 * sim.L * sim.U^2))
    return scaled_force
end

# # Set initial condition
clims = (-3, 3)
sim.flow.u .= u
# sim_step!(sim, 10)
# mean_p_1 = mean(sim.flow.p .* sim.flow.μ₀[:, :, 2])
# @show mean_p_1

# plt1 = WaterLily.flood(sim.flow.p .* sim.flow.μ₀[:, :, 2],
#                 border=:none, colorbar=true, framestyle=:none,
#                 axis=nothing, ticks=false, 
#                 clims=clims,
#                 aspect_ratio=:equal,
#                 title="time: $(sim_time(sim))",
#                 titlefontsize=12)

# # Annotate first plot with mean pressure
# annotate!(plt1, 
#     5, 15,
#     text("⟨p⟩ = $(round(mean_p_1, digits=4))", :black, 10, :left, :top),
#     subplot=1
# )

# push!(sim.flow.Δt, 0.01)
# sim_step!(sim)
# time = 0.01*sim.U/sim.L
# mean_p_2 = mean(sim.flow.p .* sim.flow.μ₀[:, :, 2])
# @show mean_p_2

# plt2 = WaterLily.flood(sim.flow.p .* sim.flow.μ₀[:, :, 2],
#                 border=:none, colorbar=true, framestyle=:none,
#                 axis=nothing, ticks=false, 
#                 clims=clims,
#                 aspect_ratio=:equal,
#                 title="time: $(time)",
#                 titlefontsize=12)

# # Annotate second plot with mean pressure
# annotate!(plt2, 
#     5, 15,
#     text("⟨p⟩ = $(round(mean_p_2, digits=4))", :black, 10, :left, :top),
#     subplot=1
# )

# plt = plot(plt1, plt2, layout=(1, 2), size=(800, 400))
# display(plt)

# @show mean(sim.flow.p)
# @show mean(sim.flow.p.* sim.flow.μ₀[:, :, 1])# push!(sim.flow.Δt, 5)
# # sim_step!(sim)
# # sim_step!(sim, 5; verbose=false)
# push!(sim.flow.Δt, 0.01)
# sim_step!(sim)
# push!(sim.flow.Δt, 0.01)
# sim_step!(sim)
# display(WaterLily.flood(sim.flow.p .* sim.flow.μ₀[:, :, 2], clims=clims))
# @show mean(sim.flow.p.* sim.flow.μ₀[:, :, 1])



pressure = []
times = []
forces = []
save_interval = 0.05
next_save = save_interval
t_run = 10
t_extr = t_run + 5
while sim_time(sim) < 10
    push!(sim.flow.Δt, WaterLily.CFL(sim.flow))
    sim_step!(sim)
    if sim_time(sim) > next_save
        sim_info(sim)

        p = mean(sim.flow.p .* sim.flow.μ₀[:, :, 1])
        @show p
        # Record force
        force = get_forces(sim)
        push!(forces, force)
        push!(pressure, p)
        push!(times, sim_time(sim))
        next_save = sim_time(sim) + save_interval
    end
end

slow_sim = deepcopy(sim)
slow_pressure = []
slow_times = []
slow_forces = []
next_save_slow = deepcopy(next_save)
while sim_time(slow_sim) < t_extr
    push!(slow_sim.flow.Δt, 0.01)
    sim_step!(slow_sim)

    if sim_time(slow_sim) > next_save_slow
        sim_info(slow_sim)

        p = mean(slow_sim.flow.p .* slow_sim.flow.μ₀[:, :, 1])
        @show p
        # Record force
        force = get_forces(slow_sim)
        push!(slow_forces, force)
        push!(slow_pressure, p)
        push!(slow_times, sim_time(slow_sim))
        next_save_slow = sim_time(slow_sim) + save_interval
    end
end

slower_sim = deepcopy(sim)
slower_pressure = []
slower_times = []
slower_forces = []
next_save_slower = deepcopy(next_save)
while sim_time(slower_sim) < t_extr
    push!(slower_sim.flow.Δt, 0.05)
    sim_step!(slower_sim)

    if sim_time(slower_sim) > next_save_slower
        sim_info(slower_sim)

        p = mean(slower_sim.flow.p .* slower_sim.flow.μ₀[:, :, 1])
        @show p
        # Record force
        force = get_forces(slower_sim)
        push!(slower_forces, force)
        push!(slower_pressure, p)
        push!(slower_times, sim_time(slower_sim))
        next_save_slower = sim_time(slower_sim) + save_interval
    end
end

# @show sim_time(sim)

while sim_time(sim) < t_extr
    push!(sim.flow.Δt, WaterLily.CFL(sim.flow))
    sim_step!(sim)

    if sim_time(sim) > next_save
        sim_info(sim)

        p = mean(sim.flow.p .* sim.flow.μ₀[:, :, 1])
        @show p
        # Record force
        force = get_forces(sim)
        push!(forces, force)
        push!(pressure, p)
        push!(times, sim_time(sim))
        next_save = sim_time(sim) + save_interval
    end
end

# Extract drag and lift
drag = first.(forces)
lift = last.(forces)

# Plot 1: Forces over time
plt_forces = plot(
    times, drag;
    label="Drag (CFL controlled)",
    color=:red,
    linewidth=1,
    xlabel="tU/L",
    ylabel="Force coefficient",
    title="Drag and Lift Coefficients",
    framestyle=:box,
    legend_position=:topleft, 
    dpi=150,
    size=(700, 400)
)
plot!(plt_forces, times, lift; label="Lift (CFL controlled)", color=:blue, linewidth=1)
slow_drag = first.(slow_forces)
slow_lift = last.(slow_forces)
plot!(plt_forces, slow_times, slow_lift; label="Lift (Δt=0.01)",lw=2, color=:blue, ls=:dashdot, linewidth=2)
plot!(plt_forces, slow_times, slow_drag; label="Drag (Δt=0.01)",lw=2, color=:red, ls=:dashdot, linewidth=2)

slower_drag = first.(slower_forces)
slower_lift = last.(slower_forces)
plot!(plt_forces, slower_times, slower_lift; label="Lift (Δt=0.05)",lw=2, color=:blue, ls=:dot, linewidth=2)
plot!(plt_forces, slower_times, slower_drag; label="Drag (Δt=0.05)",lw=2, color=:red, ls=:dot, linewidth=2)

# hline!(plt_forces, [stats.drag_mean]; label="⟨Drag⟩", color=:darkred, linestyle=:dash, linewidth=2)

# Plot 2: Mean pressure over time
plt_pressure = plot(
    times, pressure;
    label="Mean pressure (CFL controlled)",
    color=:green,
    linewidth=1,
    xlabel="tU/L",
    ylabel="Mean pressure",
    title="Mean Pressure Over Time",
    framestyle=:box,
    dpi=150,
    size=(700, 400)
)
plot!(
    slow_times, slow_pressure;
    ls=:dashdot,
    label="Mean pressure (Δt=0.01)",
    color=:green,
    linewidth=2,
)
plot!(
    slower_times, slower_pressure;
    ls=:dot,
    label="Mean pressure (Δt=0.05)",
    color=:green,
    linewidth=2,
)


# Add vertical line and shaded regions
vline!(plt_forces, [t_run]; label="Switch to Δt=0.01", color=:black, linestyle=:dash, linewidth=2)
# vspan!(plt_forces, [0, t_run]; label="CFL step", alpha=0.15, fillcolor=:blue)
# vspan!(plt_forces, [t_run, t_extr]; label="", alpha=0.15, fillcolor=:orange)

vline!(plt_pressure, [t_run]; label="Switch to Δt=0.01", color=:black, linestyle=:dash, linewidth=2)
# vspan!(plt_pressure, [0, t_run]; label="test", alpha=0.15, color=:blue)
# vspan!(plt_pressure, [t_run, t_extr]; label="", alpha=0.15, color=:orange)

# Combine plots
plt_combined = plot(plt_forces, plt_pressure; layout=(2, 1), size=(700, 700))
display(plt_combined)
# plot!(times, mean_pressures_nobod, 
#     label="Mean pressure (zero body)",
#     color=:black,
#     linewidth=1.5,
# )
# Combine plots
# plt_combined = plot(plt_forces, plt_pressure; layout=(2, 1), size=(700, 700))

# display(plt_combined)
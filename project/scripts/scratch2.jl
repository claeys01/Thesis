using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Plots

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

t_end = 50.0          # simulation end time (tU/L)
save_interval = 0.05  # interval for updating mean flow (tU/L)

# ═══════════════════════════════════════════════════════════════════════════════
# Initialize simulation and load snapshot
# ═══════════════════════════════════════════════════════════════════════════════

sim = circle_shedding_biot(; mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

# Load data
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)
simdata = load_simdata(aenode.ae_args.full_data_path)

# Load snapshot at index 1
random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

# Set initial condition
sim.flow.u .= u
sim.flow.p .= 0

# append!(sim.flow.Δt, simdata.Δt[1:random_int-1])
# sim_step!(sim)

# Initialize mean flow tracker
sim_meanflow = MeanFlow(sim.flow; uu_stats=true)

# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

function get_forces(sim::BiotSimulation)
    raw_force = WaterLily.pressure_force(sim)
    scaled_force = Float32.(raw_force ./ (0.5 * sim.L * sim.U^2))
    return scaled_force
end

function force_stats(forces::Vector{Vector{Float32}})
    drag = first.(forces)
    lift = last.(forces)
    drag_mean = mean(drag)
    lift_rms = sqrt(mean(lift .^ 2))
    return (drag_mean=drag_mean, lift_rms=lift_rms)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run simulation
# ═══════════════════════════════════════════════════════════════════════════════

forces = Vector{Vector{Float32}}()
times = Float32[]
mean_pressures = Float32[]
mean_pressures_nobod = Float32[]

next_save = save_interval
step = 1

println("Starting simulation from snapshot...")
println("="^60)

while sim_time(sim) < t_end
    # Run one simulation step
    sim_step!(sim)
    
    # Record force
    force = get_forces(sim)
    push!(forces, force)
    push!(times, Float32(sim_time(sim)))
    
    # Record mean pressure
    pressure_nobod = sim.flow.p[2:end-1, 2:end-1] .* sim.flow.μ₀[2:end-1, 2:end-1,1]
    pressure = sim.flow.p[2:end-1, 2:end-1]
    push!(mean_pressures, Float32(mean(pressure)))
    push!(mean_pressures_nobod, Float32(mean(pressure_nobod)))

    
    # Update mean flow statistics
    if sim_time(sim) > next_save
        # plt1 = WaterLily.flood(sim.flow.p[2:end-1, 2:end-1] .* sim.flow.μ₀[2:end-1, 2:end-1,2])
        # plt2 = WaterLily.flood(sim.flow.p[2:end-1, 2:end-1] .* sim.flow.μ₀[2:end-1, 2:end-1,1])
        # # plt2 = WaterLily.flood()
        # plt = plot(plt1, plt2)
        # display(plt)
        # @assert sim.flow.μ₀[2:end-1, 2:end-1,1] == sim.flow.μ₀[2:end-1, 2:end-1,2] 

        WaterLily.update!(sim_meanflow, sim.flow)
        next_save = sim_time(sim) + save_interval
        println("Step $(step): tU/L = $(round(sim_time(sim), digits=4)), " *
                "Drag = $(round(force[1], digits=4)), " *
                "Lift = $(round(force[2], digits=4)), " *
                "⟨p⟩ = $(round(mean(pressure), digits=4))")
    end
    
    step += 1
end

println("="^60)
println("Simulation complete!\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Compute statistics
# ═══════════════════════════════════════════════════════════════════════════════

# plt1 = WaterLily.flood(sim_meanflow.P[2:end-1, 2:end-1] .* sim.flow.μ₀[2:end-1, 2:end-1,1])
# display(plt1)


stats = force_stats(forces)

println("Force Statistics:")
println("  Mean drag:  $(round(stats.drag_mean, digits=5))")
println("  RMS lift:   $(round(stats.lift_rms, digits=5))")
println("  Mean pressure: $(round(mean(mean_pressures), digits=5))")
println()

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

# Extract drag and lift
drag = first.(forces)
lift = last.(forces)

# Plot 1: Forces over time
plt_forces = plot(
    times, drag;
    label="Drag",
    color=:red,
    linewidth=2,
    xlabel="tU/L",
    ylabel="Force coefficient",
    title="Drag and Lift Coefficients",
    framestyle=:box,
    dpi=150,
    size=(700, 400)
)
plot!(plt_forces, times, lift; label="Lift", color=:blue, linewidth=2)
hline!(plt_forces, [stats.drag_mean]; label="⟨Drag⟩", color=:darkred, linestyle=:dash, linewidth=2)

# Plot 2: Mean pressure over time
plt_pressure = plot(
    times, mean_pressures;
    label="Mean pressure",
    color=:green,
    linewidth=1.5,
    xlabel="tU/L",
    ylabel="Mean pressure",
    title="Mean Pressure Over Time",
    framestyle=:box,
    dpi=150,
    size=(700, 400)
)
plot!(times, mean_pressures_nobod, 
    label="Mean pressure (zero body)",
    color=:black,
    linewidth=1.5,
)
# Combine plots
plt_combined = plot(plt_forces, plt_pressure; layout=(2, 1), size=(700, 700), xlims=(0,9))

display(plt_combined)

# # Save figures
# savedir = "figs/snapshot_run/"
# if !isdir(savedir)
#     mkdir(savedir; recursive=true)
# end

# savefig(plt_forces, joinpath(savedir, "forces.png"))
# savefig(plt_pressure, joinpath(savedir, "mean_pressure.png"))
# savefig(plt_combined, joinpath(savedir, "combined.png"))

# println("Plots saved to $savedir")
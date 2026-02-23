using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

reset_timer!(to::TimerOutput)

# load aenode struct with trained neural ai models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

# 200 epochs no physics in loss func
# node_path = "data/saved_models/NODE/16/RE2500/E200_MS_Adam_250/node_params.jld2"
# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E200_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2"
# aenode = AENODE(AE_path, node_path)

# 1000 no physics in loss func
# node_path = "data/saved_models/NODE/16/RE2500/E1000_MS_Adam_250/node_params.jld2"
# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2"
# aenode = AENODE(AE_path, node_path)

# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)

# @show typeof(simdata.force), typeof(simdata.time)

# random_int = findfirst(t -> t > aenode.ae_args.t_training, simdata.time)
random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)

sim_ref = deepcopy(sim)

train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_end = 50
n_pred = 32
n_switch = 200
with_pred = true

forces_wat  = Vector{Vector{Float32}}()
time_wat   = Float32[]

forces_preds = Vector{Vector{Float32}}()
time_pred = Float32[]

forces_ref = Vector{Vector{Float32}}()
time_ref = Float32[]

function get_forces(sim::BiotSimulation)
    raw_force = WaterLily.pressure_force(sim)
    scaled_force = Float32.(raw_force./(0.5sim.L*sim.U^2)) # scale the forces!
    return scaled_force
end

function force_stats(forces::Vector{Vector{Float32}})
    drag = first.(forces)
    lift = last.(forces)
    
    drag_mean = mean(drag)
    lift_rms = sqrt(mean(lift .^ 2))
    
    return (drag_mean = drag_mean, lift_rms = lift_rms)
end


waterlily_wall_times = Float64[]
waterlily_sim_times = Float64[]       # Simulated time advanced per WaterLily step

predict_wall_times = Float64[]
predict_sim_times = Float64[]         # Simulated time advanced per prediction

hybrid_u = AbstractArray[]
hybrid_μ₀ = AbstractArray[]

# forces_ref = 

pred_idx = Int64[]
step = 2
anim = Plots.Animation()
max_original_force = maximum(maximum(abs.(f)) for f in simdata.force)

while sim_time(sim) < t_end
    if (step % n_switch == 0)
        if with_pred
            pred_Δt = 0.35f0
            # @show get_forces(sim)
            wall_time = @elapsed begin
                predict_n!(sim, aenode, n_pred; 
                Δt=pred_Δt, 
                impose_biot=true)
            end
            WaterLily.logger("data/test_psolver")

            sim_dt = n_pred * pred_Δt*sim.U/sim.L

            push!(predict_wall_times, wall_time)
            push!(predict_sim_times, sim_dt)
            
            forces = get_forces(sim)

            println(" Inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3)), wall time: $(round(wall_time*1000, digits=4)) ms, force: $forces")

            push!(forces_wat, forces)
            push!(time_wat, Float32(round(sim_time(sim),digits=4)))

            push!(forces_preds, forces)
            push!(time_pred, Float32(round(sim_time(sim),digits=4)))
            push!(pred_idx, step)
        end
    else
        wall_time = @elapsed sim_step!(sim)
        forces = get_forces(sim)

        println( "WaterLily step $step: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3)), wall time: $(round(wall_time*1000, digits=4)) ms, force: $forces")
        sim_dt = sim.flow.Δt[end]*sim.U/sim.L

        push!(waterlily_wall_times, wall_time)
        push!(waterlily_sim_times, sim_dt)
        push!(forces_wat, forces)
        push!(time_wat, Float32(round(sim_time(sim),digits=4)))

        # while sim_time(ref_sim) < sim_time(sim)
        #     sim_step(ref_sim)
        #     push!(forces_ref, forces)
        #     push!(time_ref, Float32(round(sim_time(ref_sim),digits=4)))
        #     println( "ref step $step: tU/L=$(round(sim_time(ref_sim),digits=4)), Δt=$(round(ref_sim.flow.Δt[end],digits=3)), wall time: $(round(wall_time*1000, digits=4)) ms, force: $forces")
        # end


    end
    push!(hybrid_u, sim.flow.u)
    push!(hybrid_μ₀, sim.flow.μ₀)


    step +=1
end

WaterLily.plot_logger("data/test_psolver")
savefig("figs/psolver.png")
# show(to)

# ============================================================================
# Compute Acceleration Metrics
# ============================================================================

println("\n" * "="^60)
println("ACCELERATION ANALYSIS")
println("="^60)

# Total wall-clock time spent
total_waterlily_wall = sum(waterlily_wall_times)
total_predict_wall = sum(predict_wall_times)
total_wall = total_waterlily_wall + total_predict_wall

# Average times
avg_waterlily_wall = mean(waterlily_wall_times) * 1000  # ms
avg_predict_wall = mean(predict_wall_times) * 1000      # ms
avg_waterlily_sim = mean(waterlily_sim_times)
avg_predict_sim = mean(predict_sim_times)

# Average conv
# expected_waterlily_steps = t_end/(avg_waterlily_sim) # ~ 4402
expected_waterlily_steps = 4402
estimated_pure_waterlily_time = expected_waterlily_steps * avg_waterlily_wall/1000

overall_speedup = estimated_pure_waterlily_time/total_wall

println("\n--- WaterLily Steps ---")
println("  Number of steps:     $(length(waterlily_wall_times))")
println("  Total wall time:     $(round(total_waterlily_wall, digits=3)) s")
println("  Avg wall time/step:  $(round(avg_waterlily_wall, digits=2)) ms")
println("  Avg sim time/step:   $(round(avg_waterlily_sim, digits=4)) tU/L")

println("\n--- Predictions ---")
println("  Number of predictions: $(length(predict_wall_times))")
println("  Steps per prediction:  $(n_pred)")
println("  Total wall time:       $(round(total_predict_wall, digits=3)) s")
println("  Avg wall time/pred:    $(round(avg_predict_wall, digits=2)) ms")
println("  Avg sim time/pred:     $(round(avg_predict_sim, digits=4)) tU/L")

println("\n--- Overall Comparison ---")
println("  Estimated pure WaterLily:   $(round(estimated_pure_waterlily_time, digits=2)) s")
println("  Actual hybrid time:         $(round(total_wall, digits=2)) s")
println("  Overall speedup:            $(round(overall_speedup, digits=2))x")

# For the hybrid simulation results
stats_hybrid = force_stats(forces_wat)
println("Hybrid - Drag mean: $(stats_hybrid.drag_mean), Lift RMS: $(stats_hybrid.lift_rms)")

# For the original simulation data (convert format first)
original_forces = [Float32.([d, l]) for (d, l) in simdata.force]
stats_original = force_stats(original_forces)
println("Original - Drag mean: $(stats_original.drag_mean), Lift RMS: $(stats_original.lift_rms)")

# Plot 1: Forces (existing)
plt_forces = plot(framestyle = :box, size = (600, 400), dpi = 150,
    xlabel = "tU/L", ylabel = "Force coefficient",
    xlims = (0, t_end),
    title = "Force Comparison")

plot!(plt_forces, simdata.time, first.(simdata.force), color=:red, ls=:dashdotdot, label="Original drag", alpha=0.5)
plot!(plt_forces, simdata.time, last.(simdata.force), color=:blue, ls=:dashdotdot, label="Original lift", alpha=0.5)

# hline!(plt_forces, [stats_original.drag_mean], color=:red, ls=:dash, lw=2, label="Original C̄_D", alpha=0.7)

waterlily_drag, waterlily_lift = first.(forces_wat), last.(forces_wat)
plot!(plt_forces, time_wat, waterlily_drag, label="Hybrid drag", color=:red, linewidth=1)
plot!(plt_forces, time_wat, waterlily_lift, label="Hybrid lift", color=:blue, linewidth=1)

# hline!(plt_forces, [stats_hybrid.drag_mean], color=:darkred, ls=:solid, lw=2, label="Hybrid C̄_D")

pred_drag, pred_lift = first.(forces_preds), last.(forces_preds)
scatter!(plt_forces, time_pred, pred_lift, label="prediction", color=:blue, marker=:x)

Thesis.region_spans!(plt_forces, t_train, t_test)

# display(plt_forces)
# Plot 2: Timing comparison bar chart
plt_timing = bar(
    ["WaterLily\n(per step)", "Prediction\n(per call)"],
    [avg_waterlily_wall, avg_predict_wall],
    ylabel = "Wall time (ms)",
    title = "Average Computation Time",
    legend = false,
    color = [:royalblue, :orange],
    framestyle = :box,
    size = (400, 350),
    dpi = 150,
    ylim = (0, avg_predict_wall[end] +10)

)

# Add speedup annotation
# annotate!(plt_timing, [(2, avg_predict_wall + 5, 
    # text("$(round(speedup_factor, digits=1))x faster\nper sim_time", 8, :center))])

# Plot 3: Throughput comparison
plt_total = bar(
    ["WaterLily", "Hybrid"],
    [estimated_pure_waterlily_time, total_wall],
    ylabel = "Wall time (s)",
    title = "Total Simulation Time",
    legend = false,
    color = [:royalblue, :orange],
    framestyle = :box,
    size = (400, 350),
    dpi = 150, 
    ylim = (0, maximum((estimated_pure_waterlily_time[end], total_wall[end])) + 10)
)

# Combine plots
plt_combined = plot(plt_forces, plt_timing, plt_total;
    layout = @layout([a; b c]),
    size = (800, 700))

display(plt_combined)

# base_plt_rst, _ = Thesis.plot_reynolds_stresses(simdata)
# hybrid_plt_rst, _ = Thesis.plot_reynolds_stresses(Thesis.RST(cat(hybrid_u...; dims=4), sim.flow.μ₀)...)
# rst_comp_plot = plot(base_plt_rst, hybrid_plt_rst; 
#     layout=(2 ,1), 
#     size=(1000, 500),
#     colorbar_width=1,  # Narrower colorbar
#     dpi=150)
# display(rst_comp_plot)


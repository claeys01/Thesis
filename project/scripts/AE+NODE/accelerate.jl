using Thesis
using WaterLily
using WaterLily: MeanFlow
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

random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)
sim_meanflow = MeanFlow(sim.flow; uu_stats=true)


train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_end = 50
n_pred = 32
n_switch = 100
pred_Δt = 0.35f0
with_pred = true

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

hybrid_forces_wat  = Vector{Vector{Float32}}()
hybrid_time_wat   = Float32[]

hybrid_forces_preds = Vector{Vector{Float32}}()
hybrid_time_pred = Float32[]

hybrid_waterlily_wall_times = Float64[]
hybrid_waterlily_sim_times = Float64[]       # Simulated time advanced per WaterLily step

hybrid_predict_wall_times = Float64[]
hybrid_predict_sim_times = Float64[]         # Simulated time advanced per prediction

forces_ref = Vector{Vector{Float32}}()
time_ref = Float32[]

reference_wall_times = Float64[]
reference_sim_times = Float64[]

pred_idx = Int64[]
step = 1
ref_step = 1

ref_sim = deepcopy(sim)

save_interval = 0.25 # in CTU
next_save = save_interval
ref_meanflow = MeanFlow(ref_sim.flow; uu_stats=true)

# warmup
warmup_sim = deepcopy(sim)
predict_wall_time = predict_n!(warmup_sim, aenode, n_pred; 
    Δt=pred_Δt, 
    impose_biot=true)

while sim_time(sim) < t_end
    if (step % n_switch == 0)
        if with_pred
            predict_wall_time = @elapsed begin
                predict_n!(sim, aenode, n_pred; 
                Δt=pred_Δt, 
                impose_biot=true)
            end

            sim_dt = n_pred * pred_Δt*sim.U/sim.L

            push!(hybrid_predict_wall_times, predict_wall_time)
            push!(hybrid_predict_sim_times, sim_dt)
            forces = get_forces(sim)
            push!(hybrid_forces_wat, forces)
            push!(hybrid_time_wat, Float32(round(sim_time(sim),digits=4)))
            println(" Inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3)), wall time: $(round(predict_wall_time*1000, digits=4)) ms, force: $forces")

            push!(hybrid_forces_preds, forces)
            push!(hybrid_time_pred, Float32(round(sim_time(sim),digits=4)))
            push!(pred_idx, step)
        end
    else
        hybrid_waterlily_wall_time = @elapsed sim_step!(sim)
        sim_dt = sim.flow.Δt[end]*sim.U/sim.L

        
        forces = get_forces(sim)
        push!(hybrid_forces_wat, forces)
        push!(hybrid_time_wat, Float32(round(sim_time(sim),digits=4)))
        push!(hybrid_waterlily_wall_times, hybrid_waterlily_wall_time)
        push!(hybrid_waterlily_sim_times, sim_dt)

        println( "WaterLily step $step: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3)), wall time: $(round(hybrid_waterlily_wall_time*1000, digits=4)) ms, force: $forces")

    end

    while sim_time(sim) > sim_time(ref_sim)
        reference_wall_time = @elapsed sim_step!(ref_sim)
        sim_dt = ref_sim.flow.Δt[end]*ref_sim.U/ref_sim.L

        push!(reference_wall_times, reference_wall_time)
        push!(reference_sim_times, sim_dt)

        force_ref = get_forces(ref_sim)
        push!(forces_ref, force_ref)
        push!(time_ref, Float32(round(sim_time(ref_sim),digits=4)))
        println( "* Reference step $ref_step: tU/L=$(round(sim_time(ref_sim),digits=4)), Δt=$(round(ref_sim.flow.Δt[end],digits=3)), wall time: $(round(reference_wall_time*1000, digits=4)) ms, force: $forces")
        ref_step +=1

    end

    # now sim_ref is one timestep ahead of sim

    if WaterLily.sim_time(sim) > next_save
        WaterLily.update!(sim_meanflow, sim.flow)
        WaterLily.update!(ref_meanflow, ref_sim.flow)
        next_save = sim_time(sim) + save_interval
        println("Saved mean flow statistics.")
        println("sim time:$(round(sim_time(sim),digits=4)), ref_sim time: $(round(sim_time(ref_sim),digits=4)), next save: $next_save")
    end

    step +=1
end

step = 0
ref_step = 0

# ============================================================================
# Compute Acceleration Metrics
# ============================================================================

println("\n" * "="^60)
println("ACCELERATION ANALYSIS")
println("="^60)

# Hybrid times
total_hybrid_waterlily_wall = sum(hybrid_waterlily_wall_times)
total_hybrid_predict_wall   = sum(hybrid_predict_wall_times)
total_hybrid_wall = total_hybrid_waterlily_wall + total_hybrid_predict_wall

# Average hybrid times
avg_hybrid_waterlily_wall = mean(hybrid_waterlily_wall_times) * 1000  # ms
avg_hybrid_predict_wall   = mean(hybrid_predict_wall_times)   * 1000  # ms
avg_hybrid_wall = (avg_hybrid_waterlily_wall + avg_hybrid_predict_wall) /2

avg_hybrid_waterlily_sim  = mean(hybrid_waterlily_sim_times)
avg_hybrid_predict_sim    = mean(hybrid_predict_sim_times)
avg_hybrid_sim = (avg_hybrid_waterlily_sim + avg_hybrid_predict_sim) / 2

# Reference times
total_reference_wall = sum(reference_wall_times)

# Average reference times
average_reference_wall = mean(reference_wall_times) * 1000 # ms
average_reference_sim = mean(reference_sim_times)

reference_waterlily_steps = length(reference_wall_times)

overall_speedup = total_reference_wall/total_hybrid_wall

println("\n--- Reference ---")
println("  Number of steps:     $(reference_waterlily_steps)")
println("  Total wall time:     $(total_reference_wall) s")
println("  Avg wall time/step:  $(round(average_reference_wall, digits=2)) ms")
println("  Avg sim time/step:   $(round(average_reference_sim, digits=4)) tU/L")

println("\n--- Hybrid ---")
println("  Number of steps:     $(length(hybrid_time_wat))")
println("  Total wall time:     $(total_hybrid_wall) s")
println("  Avg wall time/step:  $(round(avg_hybrid_wall, digits=2)) ms")
println("  Avg sim time/step:   $(round(avg_hybrid_sim, digits=4)) tU/L")


println("\n--- Predictions ---")
println("  Number of predictions: $(length(hybrid_predict_wall_times))")
println("  Steps per prediction:  $(n_pred)")
println("  Total wall time:       $(round(total_hybrid_predict_wall, digits=3)) s")
println("  Avg wall time/pred:    $(round(avg_hybrid_predict_wall, digits=2)) ms")
println("  Avg sim time/pred:     $(round(avg_hybrid_predict_sim, digits=4)) tU/L")

println("\n--- Overall Comparison ---")
println("  Reference WaterLily:   $(round(total_reference_wall, digits=2)) s")
println("  Actual hybrid time:    $(round(total_hybrid_wall, digits=2)) s")
println("  Overall speedup:       $(round(overall_speedup, digits=4))x")

println("\n" * "="^60)
println("FORCE ANALYSIS")
println("="^60 * "\n")
# For the hybrid simulation results
stats_hybrid = force_stats(hybrid_forces_wat)

# For the reference simulation
stats_ref = force_stats(forces_ref)
abs_err = map((x,y)->abs(x-y), stats_ref, stats_hybrid)
rel_err = map((x,y)->abs((x-y)/x)*100, stats_ref, stats_hybrid)

println("Reference - Drag mean: $(round(stats_ref.drag_mean, digits=5)),   Lift RMS: $(round(stats_ref.lift_rms, digits=5))")
println("Hybrid    - Drag mean: $(round(stats_hybrid.drag_mean, digits=5)),   Lift RMS: $(round(stats_hybrid.lift_rms, digits=5))")
println("-"^60)
println("Abs Err   - Drag mean:  $(round(abs_err.drag_mean, digits=5)),   Lift RMS: $(round(abs_err.lift_rms, digits=5))")
println("Rel Err   - Drag mean:  $(round(rel_err.drag_mean, digits=5)) %, Lift RMS: $(round(rel_err.lift_rms, digits=5)) %")
println("\n" * "="^60)

# Plot 1: Forces (existing)
plt_forces = plot(framestyle = :box, size = (600, 400), dpi = 500,
    xlabel = "tU/L", ylabel = "Force coefficient",
    xlims = (0, t_end),
    title = "Force Comparison")

ref_dag, ref_lift = first.(forces_ref), last.(forces_ref)
plot!(plt_forces, time_ref, ref_dag, color=:red, ls=:dashdotdot, label="Reference drag", alpha=0.5)
plot!(plt_forces, time_ref, ref_lift, color=:blue, ls=:dashdotdot, label="Refence lift", alpha=0.5)

# plot!(plt_forces, simdata.time, first.(simdata.force), color=:red, ls=:dash, label="Database drag", alpha=0.5)
# plot!(plt_forces, simdata.time, last.(simdata.force), color=:blue, ls=:dash, label="Database lift", alpha=0.5)
# hline!(plt_forces, [stats_original.drag_mean], color=:red, ls=:dash, lw=2, label="Original C̄_D", alpha=0.7)

waterlily_drag, waterlily_lift = first.(hybrid_forces_wat), last.(hybrid_forces_wat)
plot!(plt_forces, hybrid_time_wat, waterlily_drag, label="Hybrid drag", color=:red, linewidth=1)
plot!(plt_forces, hybrid_time_wat, waterlily_lift, label="Hybrid lift", color=:blue, linewidth=1)

# hline!(plt_forces, [stats_hybrid.drag_mean], color=:darkred, ls=:solid, lw=2, label="Hybrid C̄_D")

pred_drag, pred_lift = first.(hybrid_forces_preds), last.(hybrid_forces_preds)
# scatter!(plt_forces, hybrid_time_pred, pred_lift, label="prediction", color=:blue, marker=:x, markersize=4, markeralpha=1)
labeled = false
for i in pred_idx
    range = i-1:i
    plot!(plt_forces, hybrid_time_wat[range], waterlily_lift[range], 
        label = labeled ? "" : "Prediction",
        color=:black, 
        lw=2, 
        marker=:circle, 
        markersize=2, 
        markerstrokewidth=1)
    labeled = true
end

Thesis.region_spans!(plt_forces, t_train, t_test)

# Plot 2: Timing comparison bar chart
plt_timing = bar(
    ["WaterLily\n(per step)", "Prediction\n(per call)"],
    [average_reference_wall, avg_hybrid_predict_wall],
    ylabel = "Wall time (ms)",
    title = "Average Computation Time",
    legend = false,
    color = [:royalblue, :orange],
    framestyle = :box,
    size = (400, 350),
    dpi = 500,
    ylim = (0, avg_hybrid_predict_wall[end] +10)

)

# Plot 3: Throughput comparison
plt_total = bar(
    ["WaterLily", "Hybrid"],
    [total_reference_wall, total_hybrid_wall],
    ylabel = "Wall time (s)",
    title = "Total Simulation Time",
    legend = false,
    color = [:royalblue, :orange],
    framestyle = :box,
    size = (400, 350),
    dpi = 500, 
    ylim = (0, maximum((total_reference_wall[end], total_hybrid_wall[end])) + 10)
)

# Combine plots
plt_combined = plot(plt_forces, plt_timing, plt_total;
    layout = @layout([a; b c]),
    size = (800, 700))


function rst_plot(rst_term, clims)
    WaterLily.flood(rst_term; 
        levels=20,
        color=:viridis,
        aspectratio=:equal, 
        # border=:none, 
        # framestyle=:none,
        clims=clims,
        axis=nothing, 
        colorbar=true,
        xlims=(0, size(rst_term)[1]),
        ylims=(0, size(rst_term)[1]),
        size=(300,300),
    )
end

τ = uu(sim_meanflow)
τ_ref = uu(ref_meanflow)

rst_comp_plots = []
ranges = [(1, 1), (2, 2), (2, 1)]
titles = ["⟨u'u'⟩", "⟨v'v'⟩", "⟨u'v'⟩"]
for (i, (i3, i4)) in enumerate(ranges)
    τ_comp, τ_ref_comp = τ[:, :, i3, i4], τ_ref[:, :, i3, i4]

    clims = (minimum((minimum(τ_comp), minimum(τ_ref_comp))),
             maximum((maximum(τ_comp), maximum(τ_ref_comp))))

    p_ref, p_hybrid = rst_plot(τ_ref_comp, clims), plot!(rst_plot(τ_comp, clims))
    plt = plot(p_ref, p_hybrid, layout=(1, 2), 
        plot_title="$(titles[i]) (Reference vs Hybrid)", 
        top_margin=(-10, :mm),
        size=(600, 300))

    push!(rst_comp_plots, plt)
end
rst_comp_plot = plot(rst_comp_plots...,
                layout=(3, 1), 
                size=(900, 1100))

function plot_meanflow_comparison(sim_meanflow, ref_meanflow)
    # Extract mean u and v
    sim_u = sim_meanflow.U[:, :, 1]
    sim_v = sim_meanflow.U[:, :, 2]
    ref_u = ref_meanflow.U[:, :, 1]
    ref_v = ref_meanflow.U[:, :, 2]

    # Shared color limits for fair comparison
    u_clims = (min(minimum(sim_u), minimum(ref_u)), max(maximum(sim_u), maximum(ref_u)))
    v_clims = (min(minimum(sim_v), minimum(ref_v)), max(maximum(sim_v), maximum(ref_v)))

    # Plots
    plt_sim_u = flood(sim_u; clims=u_clims, color=:viridis,xlims=(0, 258), ylims=(0, 258), title="Hybrid ⟨u⟩",    colorbar=true, framestyle=:none, border=:none, xlabel="x", ylabel="y", aspectratio=:equal)
    plt_ref_u = flood(ref_u; clims=u_clims, color=:viridis,xlims=(0, 258), ylims=(0, 258), title="Reference ⟨u⟩", colorbar=true, framestyle=:none, border=:none, xlabel="x", ylabel="y", aspectratio=:equal)
    plt_sim_v = flood(sim_v; clims=v_clims, color=:viridis,xlims=(0, 258), ylims=(0, 258), title="Hybrid ⟨v⟩",    colorbar=true, framestyle=:none, border=:none, xlabel="x", ylabel="y", aspectratio=:equal)
    plt_ref_v = flood(ref_v; clims=v_clims, color=:viridis,xlims=(0, 258), ylims=(0, 258), title="Reference ⟨v⟩", colorbar=true, framestyle=:none, border=:none, xlabel="x", ylabel="y", aspectratio=:equal)

    # Combine in a 2x2 grid
    plt = plot(plt_ref_u, plt_sim_u, plt_ref_v, plt_sim_v;
        layout=(2,2), size=(900,700), dpi=150)
    # display(plt)
    return plt
end

# Call the function after your meanflow objects are updated:
plt_meanflow = plot_meanflow_comparison(sim_meanflow, ref_meanflow)

display(rst_comp_plot)
display(plt_combined)
display(plt_meanflow)

savefig(rst_comp_plot, "figs/acceleration/rst_comp_plot.png")
savefig(plt_combined, "figs/acceleration/plt_combined.png")
savefig(plt_meanflow, "figs/acceleration/plt_meanflow.png")


# nothing
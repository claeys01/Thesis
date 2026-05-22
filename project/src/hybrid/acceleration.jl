Base.@kwdef mutable struct AccelResults
    hybrid_forces_wat::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
    hybrid_time_wat::Vector{Float32} = Float32[]
    hybrid_forces_preds::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
    hybrid_time_pred::Vector{Float32} = Float32[]
    hybrid_waterlily_wall_times::Vector{Float64} = Float64[]
    hybrid_waterlily_sim_times::Vector{Float64} = Float64[]
    hybrid_predict_wall_times::Vector{Float64} = Float64[]
    hybrid_predict_sim_times::Vector{Float64} = Float64[]
    forces_ref::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
    time_ref::Vector{Float32} = Float32[]
    reference_wall_times::Vector{Float64} = Float64[]
    reference_sim_times::Vector{Float64} = Float64[]
    pred_idx::Vector{Int64} = Int64[]
    pred_ranges::Vector{UnitRange{Int64}} = UnitRange{Int64}[]
end

function force_stats(forces::Vector{Vector{Float32}})
    drag = first.(forces)
    lift = last.(forces)
    drag_mean = mean(drag)
    lift_rms = sqrt(mean(lift .^ 2))
    return (drag_mean = drag_mean, lift_rms = lift_rms)
end

function record_waterlily_step!(res::AccelResults, sim, wall_time)
    sim_dt = sim.flow.Δt[end] * sim.U / sim.L
    forces = get_forces(sim)
    push!(res.hybrid_forces_wat, forces)
    push!(res.hybrid_time_wat, Float32(round(sim_time(sim), digits=4)))
    push!(res.hybrid_waterlily_wall_times, wall_time)
    push!(res.hybrid_waterlily_sim_times, sim_dt)
end



function record_prediction!(res::AccelResults, sim, wall_time, sim_dt, step;
    pred_forces::Vector{Vector{Float32}}=Vector{Vector{Float32}}(),
    pred_times::Vector{Float32}=Float32[])
    forces = get_forces(sim)
    push!(res.hybrid_predict_wall_times, wall_time)
    push!(res.hybrid_predict_sim_times, sim_dt)
    pred_start = isempty(res.hybrid_time_wat) ? 1 : length(res.hybrid_time_wat)

    append!(res.hybrid_forces_preds, pred_forces)
    append!(res.hybrid_time_pred, pred_times)
    push!(res.hybrid_forces_preds, forces)
    push!(res.hybrid_time_pred, Float32(round(sim_time(sim), digits=4)))

    append!(res.hybrid_forces_wat, pred_forces)
    append!(res.hybrid_time_wat, pred_times)
    push!(res.hybrid_forces_wat, forces)
    push!(res.hybrid_time_wat, Float32(round(sim_time(sim), digits=4)))

    push!(res.pred_idx, length(res.hybrid_time_wat))
    push!(res.pred_ranges, Int64(pred_start):Int64(length(res.hybrid_time_wat)))
    return forces
end

function step_reference!(res::AccelResults, ref_sim)
    wall_time = @elapsed sim_step!(ref_sim)
    sim_dt = ref_sim.flow.Δt[end] * ref_sim.U / ref_sim.L
    push!(res.reference_wall_times, wall_time)
    push!(res.reference_sim_times, sim_dt)
    push!(res.forces_ref, get_forces(ref_sim))
    push!(res.time_ref, Float32(round(sim_time(ref_sim), digits=4)))
end

function compute_metrics(res::AccelResults)
    total_hybrid_waterlily_wall = sum(res.hybrid_waterlily_wall_times)
    total_hybrid_predict_wall = sum(res.hybrid_predict_wall_times)
    total_hybrid_wall = total_hybrid_waterlily_wall + total_hybrid_predict_wall

    avg_hybrid_waterlily_wall = mean(res.hybrid_waterlily_wall_times) * 1000
    avg_hybrid_predict_wall = mean(res.hybrid_predict_wall_times) * 1000
    avg_hybrid_wall = (avg_hybrid_waterlily_wall + avg_hybrid_predict_wall) / 2

    avg_hybrid_predict_sim = mean(res.hybrid_predict_sim_times)
    avg_hybrid_waterlily_sim = mean(res.hybrid_waterlily_sim_times)
    avg_hybrid_sim = (avg_hybrid_waterlily_sim + avg_hybrid_predict_sim) / 2

    total_reference_wall = sum(res.reference_wall_times)
    average_reference_wall = mean(res.reference_wall_times) * 1000
    average_reference_sim = mean(res.reference_sim_times)
    overall_speedup = total_reference_wall / total_hybrid_wall

    stats_hybrid = force_stats(res.hybrid_forces_wat)
    stats_ref = force_stats(res.forces_ref)
    abs_err = map((x, y) -> abs(x - y), stats_ref, stats_hybrid)
    rel_err = map((x, y) -> abs((x - y) / x) * 100, stats_ref, stats_hybrid)

    return (;
        total_hybrid_waterlily_wall, total_hybrid_predict_wall, total_hybrid_wall,
        avg_hybrid_waterlily_wall, avg_hybrid_predict_wall, avg_hybrid_wall,
        avg_hybrid_predict_sim, avg_hybrid_waterlily_sim, avg_hybrid_sim,
        total_reference_wall, average_reference_wall, average_reference_sim,
        overall_speedup,
        stats_hybrid, stats_ref, abs_err, rel_err,
    )
end

function print_metrics(res::AccelResults; pred_label="", avg_steps_per_pred=nothing)
    m = compute_metrics(res)

    println("\n" * "="^60)
    println("ACCELERATION ANALYSIS")
    println("="^60)

    println("\n--- Reference ---")
    println("  Number of steps:     $(length(res.reference_wall_times))")
    println("  Total wall time:     $(m.total_reference_wall) s")
    println("  Avg wall time/step:  $(round(m.average_reference_wall, digits=2)) ms")
    println("  Avg sim time/step:   $(round(m.average_reference_sim, digits=4)) tU/L")

    println("\n--- Hybrid ---")
    println("  Number of steps:     $(length(res.hybrid_time_wat))")
    println("  Total wall time:     $(m.total_hybrid_wall) s")
    println("  Avg wall time/step:  $(round(m.avg_hybrid_wall, digits=2)) ms")
    println("  Avg sim time/step:   $(round(m.avg_hybrid_sim, digits=4)) tU/L")

    println("\n--- Predictions $(pred_label) ---")
    println("  Number of predictions: $(length(res.hybrid_predict_wall_times))")
    if !isnothing(avg_steps_per_pred)
        println("  Avg steps/pred:        $(round(avg_steps_per_pred))")
    end
    println("  Total wall time:       $(round(m.total_hybrid_predict_wall, digits=3)) s")
    println("  Avg wall time/pred:    $(round(m.avg_hybrid_predict_wall, digits=2)) ms")
    println("  Avg sim time/pred:     $(round(m.avg_hybrid_predict_sim, digits=4)) tU/L")

    println("\n--- Overall Comparison ---")
    println("  Reference WaterLily:   $(round(m.total_reference_wall, digits=2)) s")
    println("  Actual hybrid time:    $(round(m.total_hybrid_wall, digits=2)) s")
    println("  Overall speedup:       $(round(m.overall_speedup, digits=4))x")

    println("\n" * "="^60)
    println("FORCE ANALYSIS")
    println("="^60 * "\n")
    println("Reference - Drag mean: $(round(m.stats_ref.drag_mean, digits=5)),   Lift RMS: $(round(m.stats_ref.lift_rms, digits=5))")
    println("Hybrid    - Drag mean: $(round(m.stats_hybrid.drag_mean, digits=5)),   Lift RMS: $(round(m.stats_hybrid.lift_rms, digits=5))")
    println("-"^60)
    println("Abs Err   - Drag mean:  $(round(m.abs_err.drag_mean, digits=5)),   Lift RMS: $(round(m.abs_err.lift_rms, digits=5))")
    println("Rel Err   - Drag mean:  $(round(m.rel_err.drag_mean, digits=5)) %, Lift RMS: $(round(m.rel_err.lift_rms, digits=5)) %")
    println("\n" * "="^60)
end

function plot_forces_comparison(res::AccelResults, t_end; t_train=nothing, t_test=nothing, mode_log=nothing)
    m = compute_metrics(res)
    rel_drag = round(m.rel_err.drag_mean, digits=2)
    rel_lift = round(m.rel_err.lift_rms, digits=2)

    plt = plot(framestyle=:box, size=(600, 300), dpi=500,
        titlefontsize=14,
        guidefontsize=12, tickfontsize=8, legendfontsize=9,
        xlabel="\$t^*\$", ylabel="Force coefficient",
        bottom_margin    = 2Plots.mm,
        xlims=(0, t_end), ylims=(-3, 2))

    if !isnothing(mode_log)
        train_labeled = false
        hybrid_labeled = false
        for log in mode_log
            if log.mode == "Training"
                vspan!(plt, [log.t_start, log.t_end]; fillcolor=:green, alpha=0.1, label=train_labeled ? "" : "Train region"); train_labeled = true
            elseif log.mode == "Hybrid"
                vspan!(plt, [log.t_start, log.t_end]; fillcolor=:blue, alpha=0.1, label=hybrid_labeled ? "" : "Hybrid region"); hybrid_labeled = true
            end
        end
    elseif !isnothing(t_train) && !isnothing(t_test)
        Thesis.region_spans!(plt, t_train, t_test)
    end

    ref_drag, ref_lift = first.(res.forces_ref), last.(res.forces_ref)
    plot!(plt, res.time_ref, ref_drag, color=:red, alpha=0.5, ls=:dashdot, label="Reference")
    plot!(plt, res.time_ref, ref_lift, color=:blue, alpha=0.5, ls=:dashdot, label="")

    wat_drag, wat_lift = first.(res.hybrid_forces_wat), last.(res.hybrid_forces_wat)
    plot!(plt, res.hybrid_time_wat, wat_drag, label="Hybrid", color=:red, linewidth=1)
    plot!(plt, res.hybrid_time_wat, wat_lift, label="", color=:blue, linewidth=1)

    labeled = false
    for rng in res.pred_ranges
        plot!(plt, res.hybrid_time_wat[rng], wat_lift[rng],
            label=labeled ? "" : "Prediction",
            color=:black, lw=1.5)
        plot!(plt, res.hybrid_time_wat[rng], wat_drag[rng],
            label="", color=:black, lw=1.5)
        labeled = true
    end

    plot!(plt, legend=:topleft, legendcolumns=1)
    return plt
end

function plot_timing_bars(res::AccelResults)
    m = compute_metrics(res)

    plt_timing = bar(
        ["WaterLily\n(per step)", "Prediction\n(per call)"],
        [m.average_reference_wall, m.avg_hybrid_predict_wall],
        ylabel="Wall time (ms)", title="Average Computation Time",
        legend=false, color=[:steelblue, :darkorange],
        framestyle=:box, size=(400, 350), dpi=500,
        ylim=(0, m.avg_hybrid_predict_wall + 10))

    y_max = max(m.total_reference_wall, m.total_hybrid_wall)
    plt_total = bar(
        ["WaterLily", "Hybrid"],
        [m.total_reference_wall, m.total_hybrid_wall],
        ylabel="Wall time (s)", title="Total Simulation Time",
        legend=false, color=[:steelblue, :darkorange],
        framestyle=:box, size=(400, 350), dpi=500,
        ylim=(0, y_max + 10))
    # annotate!(plt_total, 2, m.total_hybrid_wall + 0.05 * y_max,
        # text("$(round(m.overall_speedup, digits=2))× faster", :black, 10, :center))

    return plt_timing, plt_total
end

function plot_accel_combined(res::AccelResults, t_end; t_train=nothing, t_test=nothing, mode_log=nothing)
    plt_forces = plot_forces_comparison(res, t_end; t_train=t_train, t_test=t_test, mode_log=mode_log)
    plt_timing, plt_total = plot_timing_bars(res)
    return plot(plt_forces, plt_timing, plt_total;
        layout=@layout([a{0.6h}; b c]), size=(800, 700))
end

function rst_plot(rst_term, clims)
    WaterLily.flood(rst_term;
        levels=20, color=:viridis, aspectratio=:equal,
        clims=clims, axis=nothing, colorbar=true,
        xlims=(0, size(rst_term)[1]), ylims=(0, size(rst_term)[1]),
        size=(300, 300))
end

function plot_rst_comparison(sim_meanflow, ref_meanflow)
    τ = WaterLily.uu(sim_meanflow)
    τ_ref = WaterLily.uu(ref_meanflow)

    rst_comp_plots = []
    ranges = [(1, 1), (2, 2), (2, 1)]
    titles = ["⟨u'u'⟩", "⟨v'v'⟩", "⟨u'v'⟩"]
    for (i, (i3, i4)) in enumerate(ranges)
        τ_comp, τ_ref_comp = τ[:, :, i3, i4], τ_ref[:, :, i3, i4]
        clims = (min(minimum(τ_comp), minimum(τ_ref_comp)),
                 max(maximum(τ_comp), maximum(τ_ref_comp)))
        p_ref = rst_plot(τ_ref_comp, clims)
        p_hybrid = plot!(rst_plot(τ_comp, clims))
        plt = plot(p_ref, p_hybrid, layout=(1, 2),
            plot_title="$(titles[i]) (Reference vs Hybrid)",
            top_margin=(-10, :mm), size=(600, 300))
        push!(rst_comp_plots, plt)
    end
    return plot(rst_comp_plots..., layout=(3, 1), size=(900, 1100))
end

function plot_meanflow_comparison(sim_meanflow, ref_meanflow)
    sim_u, sim_v = sim_meanflow.U[:, :, 1], sim_meanflow.U[:, :, 2]
    ref_u, ref_v = ref_meanflow.U[:, :, 1], ref_meanflow.U[:, :, 2]

    u_clims = (min(minimum(sim_u), minimum(ref_u)), max(maximum(sim_u), maximum(ref_u)))
    v_clims = (min(minimum(sim_v), minimum(ref_v)), max(maximum(sim_v), maximum(ref_v)))

    common = (color=:viridis, xlims=(0, 258), ylims=(0, 258), colorbar=true,
              framestyle=:none, border=:none, xlabel="x", ylabel="y", aspectratio=:equal)

    plt_ref_u = flood(ref_u; title="Reference ⟨u⟩", clims=u_clims, common...)
    plt_sim_u = flood(sim_u; title="Hybrid ⟨u⟩", clims=u_clims, common...)
    plt_ref_v = flood(ref_v; title="Reference ⟨v⟩", clims=v_clims, common...)
    plt_sim_v = flood(sim_v; title="Hybrid ⟨v⟩", clims=v_clims, common...)

    return plot(plt_ref_u, plt_sim_u, plt_ref_v, plt_sim_v;
        layout=(2, 2), size=(900, 700), dpi=150)
end

function save_velocity_frame!(gif_frames::Vector, sim, time_step)
    plt_combined, _ = velocity_flood(sim)
    plt_frame = plot(plt_combined,
        plot_title="Velocity Field at tU/L = $(round(time_step, digits=3))",
        plot_titlefontsize=14)
    push!(gif_frames, plt_frame)
end

function create_velocity_gif(gif_frames::Vector, savedir::String)
    isdir(savedir) || mkdir(savedir)
    gif_path = joinpath(savedir, "velocity_evolution.gif")
    anim = Plots.Animation()
    for f in gif_frames
        frame(anim, f)
    end
    gif(anim, gif_path; fps=5, show_msg=false)
    println("GIF saved to: $gif_path")
    return gif_path
end

function save_accel_plots(savedir, plt_combined, rst_comp_plot, plt_meanflow)
    isdir(savedir) || mkdir(savedir)
    savefig(plt_combined, joinpath(savedir, "plt_combined.png"))
    savefig(rst_comp_plot, joinpath(savedir, "rst_comp_plot.png"))
    savefig(plt_meanflow, joinpath(savedir, "plt_meanflow.png"))
end

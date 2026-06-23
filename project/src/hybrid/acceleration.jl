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

function region_force_errors(res::AccelResults, mode_log)
    isnothing(mode_log) && return nothing
    spans = [(log.t_start, log.t_end) for log in mode_log if log.mode == "Hybrid"]
    isempty(spans) && return nothing
    in_region(t) = any(s -> s[1] <= t <= s[2], spans)

    hyb_forces = res.hybrid_forces_wat[findall(in_region, res.hybrid_time_wat)]
    ref_forces = res.forces_ref[findall(in_region, res.time_ref)]
    (isempty(hyb_forces) || isempty(ref_forces)) && return nothing

    stats_hybrid = force_stats(hyb_forces)
    stats_ref = force_stats(ref_forces)
    abs_err = map((x, y) -> abs(x - y), stats_ref, stats_hybrid)
    rel_err = map((x, y) -> abs((x - y) / x) * 100, stats_ref, stats_hybrid)
    return (; stats_hybrid, stats_ref, abs_err, rel_err)
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
    wall_time = @elapsed begin
        sim_step!(ref_sim)
        sync_device!()
    end
    sim_dt = ref_sim.flow.Δt[end] * ref_sim.U / ref_sim.L
    push!(res.reference_wall_times, wall_time)
    push!(res.reference_sim_times, sim_dt)
    push!(res.forces_ref, get_forces(ref_sim))
    push!(res.time_ref, Float32(round(sim_time(ref_sim), digits=4)))
end

function compute_metrics(res::AccelResults)
    safediv(a, b) = b == 0 ? 0.0 : a / b
    safemean(x) = isempty(x) ? 0.0 : mean(x)

    total_hybrid_waterlily_wall = sum(res.hybrid_waterlily_wall_times)
    total_hybrid_predict_wall = sum(res.hybrid_predict_wall_times)
    total_hybrid_wall = total_hybrid_waterlily_wall + total_hybrid_predict_wall

    total_hybrid_waterlily_sim = sum(res.hybrid_waterlily_sim_times)
    total_hybrid_predict_sim = sum(res.hybrid_predict_sim_times)
    total_hybrid_sim = total_hybrid_waterlily_sim + total_hybrid_predict_sim

    total_reference_wall = sum(res.reference_wall_times)
    total_reference_sim = sum(res.reference_sim_times)

    # Wall time to advance one convective time unit (ms / CTU)
    ref_wall_per_ctu = safediv(total_reference_wall, total_reference_sim) * 1000
    hybrid_waterlily_wall_per_ctu = safediv(total_hybrid_waterlily_wall, total_hybrid_waterlily_sim) * 1000
    hybrid_predict_wall_per_ctu = safediv(total_hybrid_predict_wall, total_hybrid_predict_sim) * 1000
    hybrid_wall_per_ctu = safediv(total_hybrid_wall, total_hybrid_sim) * 1000

    # Per-event averages, kept for the per-step / per-call breakdown in print_metrics
    avg_hybrid_waterlily_wall = safemean(res.hybrid_waterlily_wall_times) * 1000
    avg_hybrid_predict_wall = safemean(res.hybrid_predict_wall_times) * 1000
    avg_hybrid_predict_sim = safemean(res.hybrid_predict_sim_times)
    avg_hybrid_waterlily_sim = safemean(res.hybrid_waterlily_sim_times)
    average_reference_wall = safemean(res.reference_wall_times) * 1000
    average_reference_sim = safemean(res.reference_sim_times)

    overall_speedup = safediv(total_reference_wall, total_hybrid_wall)
    ctu_speedup = safediv(ref_wall_per_ctu, hybrid_wall_per_ctu)

    stats_hybrid = force_stats(res.hybrid_forces_wat)
    stats_ref = force_stats(res.forces_ref)
    abs_err = map((x, y) -> abs(x - y), stats_ref, stats_hybrid)
    rel_err = map((x, y) -> abs((x - y) / x) * 100, stats_ref, stats_hybrid)

    return (;
        total_hybrid_waterlily_wall, total_hybrid_predict_wall, total_hybrid_wall,
        total_hybrid_waterlily_sim, total_hybrid_predict_sim, total_hybrid_sim,
        total_reference_wall, total_reference_sim,
        ref_wall_per_ctu, hybrid_waterlily_wall_per_ctu,
        hybrid_predict_wall_per_ctu, hybrid_wall_per_ctu,
        avg_hybrid_waterlily_wall, avg_hybrid_predict_wall,
        avg_hybrid_predict_sim, avg_hybrid_waterlily_sim,
        average_reference_wall, average_reference_sim,
        overall_speedup, ctu_speedup,
        stats_hybrid, stats_ref, abs_err, rel_err,
    )
end

function meanflow_errors(sim_meanflow, ref_meanflow)
    MAE(a, b) = mean(abs, a .- b)
    stats(a, b) = (l1 = MAE(a, b), ρ = cor(vec(a), vec(b)))

    sim_u, sim_v = sim_meanflow.U[:, :, 1], sim_meanflow.U[:, :, 2]
    ref_u, ref_v = ref_meanflow.U[:, :, 1], ref_meanflow.U[:, :, 2]
    u  = stats(sim_u, ref_u)
    v  = stats(sim_v, ref_v)

    τ     = WaterLily.uu(sim_meanflow)
    τ_ref = WaterLily.uu(ref_meanflow)
    uu = stats(τ[:, :, 1, 1], τ_ref[:, :, 1, 1])
    vv = stats(τ[:, :, 2, 2], τ_ref[:, :, 2, 2])
    uv = stats(τ[:, :, 2, 1], τ_ref[:, :, 2, 1])

    return (; u, v, uu, vv, uv)
end

function print_metrics(res::AccelResults; pred_label="", avg_steps_per_pred=nothing,
    sim_meanflow=nothing, ref_meanflow=nothing, mode_log=nothing)
    m = compute_metrics(res)

    println("\n" * "="^60)
    println("ACCELERATION ANALYSIS")
    println("="^60)

    println("\n--- Reference ---")
    println("  Number of steps:     $(length(res.reference_wall_times))")
    println("  Total wall time:     $(m.total_reference_wall) s")
    println("  Avg wall time/step:  $(round(m.average_reference_wall, digits=2)) ms")
    println("  Wall time / CTU:     $(round(m.ref_wall_per_ctu, digits=2)) ms/CTU")

    println("\n--- Hybrid ---")
    println("  Number of steps:     $(length(res.hybrid_time_wat))")
    println("  Total wall time:     $(m.total_hybrid_wall) s")
    println("  Wall time / CTU:     $(round(m.hybrid_wall_per_ctu, digits=2)) ms/CTU")
    println("    └ CFD steps:       $(round(m.hybrid_waterlily_wall_per_ctu, digits=2)) ms/CTU")
    println("    └ Rollout:         $(round(m.hybrid_predict_wall_per_ctu, digits=2)) ms/CTU")

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
    println("  Speedup (per CTU):     $(round(m.ctu_speedup, digits=4))x")

    println("\n" * "="^60)
    println("FORCE ANALYSIS")
    println("="^60 * "\n")
    println("Reference - Drag mean: $(round(m.stats_ref.drag_mean, digits=5)),   Lift RMS: $(round(m.stats_ref.lift_rms, digits=5))")
    println("Hybrid    - Drag mean: $(round(m.stats_hybrid.drag_mean, digits=5)),   Lift RMS: $(round(m.stats_hybrid.lift_rms, digits=5))")
    println("-"^60)
    println("Abs Err   - Drag mean:  $(round(m.abs_err.drag_mean, digits=5)),   Lift RMS: $(round(m.abs_err.lift_rms, digits=5))")
    println("Rel Err   - Drag mean:  $(round(m.rel_err.drag_mean, digits=5)) %, Lift RMS: $(round(m.rel_err.lift_rms, digits=5)) %")

    region = region_force_errors(res, mode_log)
    if !isnothing(region)
        println("-"^60)
        println("(Hybrid regions only)")
        println("Reference - Drag mean: $(round(region.stats_ref.drag_mean, digits=5)),   Lift RMS: $(round(region.stats_ref.lift_rms, digits=5))")
        println("Hybrid    - Drag mean: $(round(region.stats_hybrid.drag_mean, digits=5)),   Lift RMS: $(round(region.stats_hybrid.lift_rms, digits=5))")
        println("Abs Err   - Drag mean:  $(round(region.abs_err.drag_mean, digits=5)),   Lift RMS: $(round(region.abs_err.lift_rms, digits=5))")
        println("Rel Err   - Drag mean:  $(round(region.rel_err.drag_mean, digits=5)) %, Lift RMS: $(round(region.rel_err.lift_rms, digits=5)) %")
    end

    if !isnothing(sim_meanflow) && !isnothing(ref_meanflow)
        fe = meanflow_errors(sim_meanflow, ref_meanflow)
        println("\n" * "="^60)
        println("FIELD ANALYSIS")
        println("="^60 * "\n")
        println("MAE - Mean flow ⟨u⟩: $(round(fe.u.l1, digits=3)),  ⟨v⟩: $(round(fe.v.l1, digits=3))")
        println("MAE - RST ⟨u'u'⟩: $(round(fe.uu.l1, digits=3)),  ⟨v'v'⟩: $(round(fe.vv.l1, digits=3)),  ⟨u'v'⟩: $(round(fe.uv.l1, digits=3))")
        println("Corr coeff   - Mean flow ⟨u⟩: $(round(fe.u.ρ, digits=4)),  ⟨v⟩: $(round(fe.v.ρ, digits=4))")
        println("Corr coeff   - RST ⟨u'u'⟩: $(round(fe.uu.ρ, digits=4)),  ⟨v'v'⟩: $(round(fe.vv.ρ, digits=4)),  ⟨u'v'⟩: $(round(fe.uv.ρ, digits=4))")
    end

    println("\n" * "="^60)
end

function print_timing_summary(base_dir::AbstractString)
    path = isfile(base_dir) ? base_dir : joinpath(base_dir, "timing_summary.jld2")
    t = load(path)

    println("\n" * "="^60)
    println("TIMING SUMMARY")
    println("="^60)
    println("  (file: $path)")

    println("\n--- Initial training ---")
    println("  AE train:            $(t["ae_elapsed_min"]) min")
    println("  NODE train:          $(t["node_elapsed_min"]) min")

    println("\n--- Retraining ($(t["n_retrains"]) retrains) ---")
    println("  AE retrain total:    $(t["ae_retrain_elapsed_min"]) min")
    println("  NODE retrain total:  $(t["node_retrain_elapsed_min"]) min")
    for (i, r) in enumerate(t["retrain_timings"])
        println("    [$i] WL cutoff: $(r.wl_cutoff) min, AE: $(r.ae) min, NODE: $(r.node) min")
    end

    println("\n--- WaterLily ---")
    println("  Warmup:              $(t["wl_warmup_elapsed_min"]) min")
    println("  Cutoff total:        $(t["wl_cutoff_elapsed_min"]) min")
    println("  Tail:                $(t["wl_tail_elapsed_min"]) min")

    println("\n--- Totals ---")
    println("  ML total:            $(t["ml_total_min"]) min")
    println("  WaterLily total:     $(t["wl_total_min"]) min")
    println("  Grand total:         $(t["grand_total_min"]) min")
    println("\n" * "="^60)
end

function plot_forces_comparison(res::AccelResults, t_end; t_train=nothing, t_test=nothing, mode_log=nothing)
    m = compute_metrics(res)
    rel_drag = round(m.rel_err.drag_mean, digits=2)
    rel_lift = round(m.rel_err.lift_rms, digits=2)

    plt = plot(framestyle=:box, size=(800, 400), dpi=500,
        xlabel="\$t^*\$", ylabel="Force coefficient",
        titlefontsize=14,
        guidefontsize=12, tickfontsize=8, legendfontsize=6,
        foreground_color_axis  = :black,
        foreground_color_text  = :black,
        left_margin   = 3Plots.mm,
        right_margin  = 1Plots.mm,
        top_margin    = 1Plots.mm,
        bottom_margin = 2Plots.mm,
        legend=:topright,
        background_color_legend = RGBA(1, 1, 1, 0.7),
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
    plot!(plt, res.time_ref, ref_drag, color=:red, alpha=0.5, ls=:dashdot, label="\$C_D\$ (reference)")
    plot!(plt, res.time_ref, ref_lift, color=:blue, alpha=0.5, ls=:dashdot, label="\$C_L\$ (reference)")

    wat_drag, wat_lift = first.(res.hybrid_forces_wat), last.(res.hybrid_forces_wat)
    plot!(plt, res.hybrid_time_wat, wat_drag, label="\$C_D\$ (hybrid)", color=:red, linewidth=1)
    plot!(plt, res.hybrid_time_wat, wat_lift, label="\$C_L\$ (hybrid)", color=:blue, linewidth=1)

    labeled = false
    for rng in res.pred_ranges
        plot!(plt, res.hybrid_time_wat[rng], wat_lift[rng],
            label=labeled ? "" : "Rollout",
            color=:black, lw=1.5)
        plot!(plt, res.hybrid_time_wat[rng], wat_drag[rng],
            label="", color=:black, lw=1.5)
        labeled = true
    end

    plot!(plt, legend=:bottomleft, legendcolumns=2)
    return plt
end

function plot_timing_bars(res::AccelResults)
    m = compute_metrics(res)

    timing_vals = [m.ref_wall_per_ctu, m.hybrid_wall_per_ctu, m.hybrid_waterlily_wall_per_ctu,m.hybrid_predict_wall_per_ctu]
    @show size(timing_vals)
    timing_vals = [9548.4, 4018.83, 9540, m.hybrid_predict_wall_per_ctu]
    @show size(timing_vals)
    # Reference sits apart; the three hybrid bars are spaced 1 unit apart with
    # bar_width=1 so their edges meet (touch) while keeping a gap to Reference.
    xs = [1.0, 2.5, 3.5, 4.5]
    plt_timing = bar(
        xs, timing_vals,
        ylabel="Wall time (ms)", title="Mean Cost per CTU",
        legend=false, color=[:steelblue, :darkorange, :firebrick, :seagreen],
        bar_width=1.0,
        # bar_width=bar_widths,
        alpha=[1, 1, 0.75, 0.75],
        titlefontsize=14,
        guidefontsize=12, tickfontsize=10, legendfontsize=9,
        framestyle=:box, size=(400, 350), dpi=500,
        xticks=(xs, ["Reference", "Hybrid\n(total)", "Hybrid\n(CFD)", "Hybrid\n(rollout)"]),
        # xlim=(0.3, 5.2),
        ylim=(0, maximum(timing_vals) * 1.15 + eps()))

    # The rollout bar is near-zero height, so label its value just above the baseline.
    annotate!(plt_timing, xs[4], timing_vals[4] + 0.05 * maximum(timing_vals),
        text("$(round(timing_vals[4], digits=1)) ms", :black, 8, :center))

    # y_max = max(m.total_reference_wall, m.total_hybrid_wall)
    y_max = max(954.84, 401,083)
    plt_total = bar(
        ["Reference", "Hybrid"],
        # [m.total_reference_wall, m.total_hybrid_wall],
        [954.84, 401.083],
        ylabel="Wall time (s)", title="Total Simulation Time",
        legend=false, color=[:steelblue, :darkorange],
        titlefontsize=14,
        guidefontsize=12, tickfontsize=10, legendfontsize=9,
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

# Single RST contour panel. Geometry, axes, fonts and cylinder match
# `meanflow_contour` so the RST and mean-flow panels look identical. Two styles:
#   :bw     — black-and-white line contours in the style of the reference RST
#             figure (Font García et al., fig. 11): solid lines for positive
#             levels, dashed for negative, fixed spacing Δ = clims[2]/nlevels.
#   :filled — coloured filled contours like the mean-flow plots, with the
#             contour edges drawn in thin black so the bands stay clearly visible.
function rst_plot(field; clims, nlevels=12, style=:bw,
                  cmap=cgrad(:curl), title="", colorbar=false)
    nx, ny = size(field)
    ni = nx - 2
    L  = ni / 8
    cx = ni / 4 + 1.5
    cy = (ny - 2) / 2 + 1.5
    xc = (collect(axes(field, 1)) .- cx) ./ L
    yc = (collect(axes(field, 2)) .- cy) ./ L
    xticks = (ceil(xc[1] / 2) * 2):2:(floor(xc[end] / 2) * 2)
    yticks = (ceil(yc[1] / 2) * 2):2:(floor(yc[end] / 2) * 2)

    # Keep the plot width at the default 600 px (so the plotted region stays the
    # same size) but shrink the canvas height to hug the wide-short domain
    # instead of padding it with empty bands above/below.
    panel_w = 500
    panel_h = round(Int, panel_w * (yc[end] - yc[1]) / (xc[end] - xc[1])) + 70
    colorbar && (panel_w += 90)

    axis_kw = (; aspect_ratio=:equal, framestyle=:box, legend=false,
        background=:white, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
        xticks=xticks, yticks=yticks, tickfontsize=10, tick_direction=:iout,
        xlabel="x/L", ylabel="y/L", guidefontsize=16,
        title=title, titlefontsize=10, size=(panel_w, panel_h),
        left_margin=1Plots.mm, right_margin=3Plots.mm,
        top_margin=0Plots.mm, bottom_margin=1Plots.mm)

    if style == :filled
        f = clamp.(field, clims[1], clims[2])' |> Array
        plt = contourf(xc, yc, f; levels=nlevels, color=cmap, clims=clims,
            linewidth=0.4, linecolor=:black, colorbar=colorbar, axis_kw...)
    else
        plt = plot(; colorbar=colorbar, axis_kw...)
        amax = clims[2]
        Δ = amax / nlevels
        lmin = Δ
        f = field' |> Array
        neg_levels = collect(-amax:Δ:-lmin)   # dashed
        pos_levels = collect(lmin:Δ:amax)     # solid
        if !isempty(neg_levels)
            contour!(plt, xc, yc, f; levels=neg_levels, color=:black,
                linecolor=:black, linewidth=0.6, linestyle=:dash, colorbar=false)
        end
        if !isempty(pos_levels)
            contour!(plt, xc, yc, f; levels=pos_levels, color=:black,
                linecolor=:black, linewidth=0.6, linestyle=:solid, colorbar=false)
        end
    end

    θ = range(0, 2π; length=120)
    plot!(plt, Plots.Shape(0.5 .* cos.(θ), 0.5 .* sin.(θ));
        seriestype=:shape, fillcolor="#BFBFBF", linecolor=:black, linewidth=1.0)
    return plt
end

# style = :bw for the reference-figure line contours, :filled for coloured
# filled contours. The filled colour maps are deliberately different from the
# mean-flow plots (:RdBu / :thermal): sequential :viridis for the non-negative
# normal stresses ⟨u'u'⟩, ⟨v'v'⟩ and diverging :curl for the signed ⟨u'v'⟩.
function plot_rst_comparison(sim_meanflow, ref_meanflow; savedir=nothing, fmt="pdf",
                             nlevels=12, style=:bw)
    τ = WaterLily.uu(sim_meanflow)
    τ_ref = WaterLily.uu(ref_meanflow)

    ranges = [(1, 1), (2, 2), (2, 1)]
    names  = ["rst_uu", "rst_vv", "rst_uv"]
    signed = [false, false, true]
    div = cgrad(:curl)      # signed shear stress ⟨u'v'⟩
    # non-negative normal stresses ⟨u'u'⟩, ⟨v'v'⟩: the positive (upper) half of
    # :curl so 0 maps to its white centre and max to its far end, directly
    # comparable to the signed shear panel.
    seq = cgrad([div[x] for x in range(0.5, 1.0; length=128)])
    # non-negative absolute-error fields: linear, near-white at zero, in a cool
    # hue clearly distinct from the warm/magenta stress colourbars.
    err = cgrad(:amp)

    
    panels = []
    for (i, (i3, i4)) in enumerate(ranges)
        τ_comp, τ_ref_comp = τ[:, :, i3, i4], τ_ref[:, :, i3, i4]
        τ_diff = abs.(τ_comp .- τ_ref_comp)
        # Shared limits so the reference and hybrid panels are comparable.
        if signed[i]
            clims = sym_clims(τ_comp, τ_ref_comp)
            cmap = div
        else
            amax = max(maximum(τ_comp), maximum(τ_ref_comp))
            clims = (0.0, amax)
            cmap = seq
        end
        err_clims = (0.0, maximum(τ_diff))
        kw     = (; clims=clims,     nlevels=nlevels, style=style, cmap=cmap, colorbar=(style == :filled))
        err_kw = (; clims=err_clims, nlevels=nlevels, style=style, cmap=err,  colorbar=(style == :filled))
        push!(panels, ("$(names[i])_reference", rst_plot(τ_ref_comp; kw...)))
        push!(panels, ("$(names[i])_hybrid",    rst_plot(τ_comp;     kw...)))
        push!(panels, ("$(names[i])_abserror",  rst_plot(τ_diff;     err_kw...)))
    end

    if !isnothing(savedir)
        isdir(savedir) || mkpath(savedir)
        for (name, p) in panels
            savefig(p, joinpath(savedir, "$(name).$(fmt)"))
        end
        println("Saved $(length(panels)) RST panels to: $(savedir)")
    end

    return plot((p for (_, p) in panels)...; 
        layout=(3, 3), size=(1200, 1050), dpi=400, colorbar=(style == :filled))
end

# Filled red/blue contour panel in the style of the reference vorticity figure:
# discrete contour bands with thin black contour lines, no axes, white
# background, and the cylinder body drawn as a solid gray disk.
function meanflow_contour(field; clims, title="", cmap=cgrad(:RdBu, rev=true),
                          levels=12, colorbar=true, linewidth=0.25)
    nx, ny = size(field)
    # The geometry is defined in grid cells; map array indices to cylinder
    # diameters (x/L, y/L) with the cylinder centre at the origin, matching the
    # reference vorticity figure. Diameter L = ni/8 cells (radius ni/16).
    ni = nx - 2
    L  = ni / 8
    cx = ni / 4 + 1.5
    cy = (ny - 2) / 2 + 1.5
    xc = (collect(axes(field, 1)) .- cx) ./ L
    yc = (collect(axes(field, 2)) .- cy) ./ L
    # Even-spaced ticks (in diameters) so 0 lands on the cylinder centre.
    xticks = (ceil(xc[1] / 2) * 2):2:(floor(xc[end] / 2) * 2)
    yticks = (ceil(yc[1] / 2) * 2):2:(floor(yc[end] / 2) * 2)
    # Keep the plot width at the default 600 px (so the plotted region stays the
    # same size) but shrink the canvas height to hug the wide-short domain
    # instead of padding it with empty bands above/below.
    panel_w = 500
    panel_h = round(Int, panel_w * (yc[end] - yc[1]) / (xc[end] - xc[1])) + 70
    colorbar && (panel_w += 90)
    f = clamp.(field, clims[1], clims[2])
    plt = contourf(xc, yc, f' |> Array;
        levels=levels, color=cmap, clims=clims,
        linewidth=linewidth, linecolor=:black,
        aspect_ratio=:equal, framestyle=:box, legend=false,
        colorbar=colorbar, background=:white,
        xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]),
        xticks=xticks, yticks=yticks, tickfontsize=10, tick_direction=:iout,
        xlabel="x/L", ylabel="y/L", guidefontsize=16,
        title=title, titlefontsize=10, size=(panel_w, panel_h),
        left_margin=1Plots.mm, right_margin=3Plots.mm,
        top_margin=0Plots.mm, bottom_margin=1Plots.mm)
    θ = range(0, 2π; length=120)
    plot!(plt, Plots.Shape(0.5 .* cos.(θ), 0.5 .* sin.(θ));
        seriestype=:shape, fillcolor="#BFBFBF", linecolor=:black, linewidth=1.0)
    return plt
end

# Symmetric color limits around zero so 0 maps to white in the diverging map.
function sym_clims(arrays...)
    m = maximum(maximum(abs, a) for a in arrays)
    return (-m, m)
end

function plot_meanflow_comparison(sim_meanflow, ref_meanflow; savedir=nothing, fmt="pdf")
    sim_u, sim_v = sim_meanflow.U[:, :, 1], sim_meanflow.U[:, :, 2]
    ref_u, ref_v = ref_meanflow.U[:, :, 1], ref_meanflow.U[:, :, 2]
    u_diff = abs.(sim_u .- ref_u)
    v_diff = abs.(sim_v .- ref_v)

    u_clims = sym_clims(sim_u, ref_u)
    v_clims = sym_clims(sim_v, ref_v)
    u_err_clims = (0.0, maximum(u_diff))
    v_err_clims = (0.0, maximum(v_diff))

    div = cgrad(:RdBu, rev=true)   # red = positive, blue = negative
    seq = cgrad(:amp)          # non-negative absolute-error fields

    panels = [
        ("meanflow_u_reference", meanflow_contour(ref_u;  clims=u_clims,     cmap=div, title="")),
        ("meanflow_u_hybrid",    meanflow_contour(sim_u;  clims=u_clims,     cmap=div, title="")),
        ("meanflow_u_abserror",  meanflow_contour(u_diff; clims=u_err_clims, cmap=seq, title="")),
        ("meanflow_v_reference", meanflow_contour(ref_v;  clims=v_clims,     cmap=div, title="")),
        ("meanflow_v_hybrid",    meanflow_contour(sim_v;  clims=v_clims,     cmap=div, title="")),
        ("meanflow_v_abserror",  meanflow_contour(v_diff; clims=v_err_clims, cmap=seq, title="")),
    ]

    if !isnothing(savedir)
        isdir(savedir) || mkpath(savedir)
        for (name, p) in panels
            savefig(p, joinpath(savedir, "$(name).$(fmt)"))
        end
        println("Saved $(length(panels)) mean-flow panels to: $(savedir)")
    end

    return plot((p for (_, p) in panels)...;
        layout=(2, 3), size=(1200, 700), dpi=400)
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

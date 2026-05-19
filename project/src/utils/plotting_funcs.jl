function plot_reynolds_stresses(uu, vv, uv)
    # Compute clims for each component: (min, max) centered around mean
    # uu_min, uu_max = (0.0, maximum(uu))
    # vv_min, vv_max = (0.0, maximum(vv))
    uu_min, uu_max = extrema(uu)
    vv_min, vv_max = extrema(vv)
    uv_min, uv_max = extrema(uv)
    # uu_min, uu_max = 0.0, mean(uu) + std(uu)
    # vv_min, vv_max = 0.0, mean(vv) + std(vv)
    # uv_min, uv_max = mean(uv) - std(uv), mean(uv) + std(uv)
    
    
    # Plot ⟨u'u'⟩
    plt_uu = flood(uu; 
        clims=(uu_min, uu_max),
        levels=20,
        color=:viridis,
        title="⟨u'u'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal, 
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Plot ⟨v'v'⟩
    plt_vv = flood(vv;
        clims=(vv_min, vv_max),
        levels=20,
        color=:viridis,
        title="⟨v'v'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal,
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Plot ⟨u'v'⟩
    plt_uv = flood(uv;
        clims=(uv_min, uv_max),
        levels=20,
        color=:viridis,
        title="⟨u'v'⟩",
        xlabel="x",
        ylabel="y",
        aspectratio=:equal, 
        border=:none, 
        framestyle=:none,
        axis=nothing
    )
    
    # Combine into a single figure
    plt_combined = plot(plt_uu, plt_vv, plt_uv;
        layout=(1, 3),
        size=(1200, 350),
        dpi=150,
        # plot_title="Reynolds Stress Components (Re=2500)",
        # plot_titlefontsize=12
    )
    
    return plt_combined, (plt_uu, plt_vv, plt_uv)
end
plot_reynolds_stresses(simdata::SimData) = plot_reynolds_stresses(RST(simdata.u, simdata.μ₀[:, :, :, 1])...)


function train_force_plot(forces::Vector{Vector{Float32}}, time::Vector{Float32}; train_idx=nothing, val_idx=nothing, test_idx=nothing, show_zeros=true)
    drag = first.(forces)
    lift = last.(forces)
    zero_idxs = zero_crossing(lift; direction=:rising)

    plt = plot(framestyle=:box, size=(600, 300), dpi=500,
        xlabel="\$t^*\$", ylabel="Force coefficient",
        xlims=(0, 50), ylims=(-3, 2),
        titlefontsize=14,
        guidefontsize=12, tickfontsize=8, legendfontsize=7,
        foreground_color_axis  = :black,
        foreground_color_text  = :black,
        left_margin   = 3Plots.mm,
        right_margin  = 1Plots.mm,
        top_margin    = 1Plots.mm,
        bottom_margin = 2Plots.mm,
        legend=:topright) 
    plot!(plt, time, drag, label=L"C_{d}", color=:red, lw=1)
    plot!(plt, time, lift, label=L"C_{L}", color=:blue, lw=1)


    if !isnothing(val_idx) && !isempty(val_idx)
        scatter!(plt, time[val_idx], lift[val_idx], 
        markersize = 2, color=:black, markerstrokewidth = 0, markershape =:circle, 
        label="validation points")
        scatter!(plt, time[val_idx], drag[val_idx], 
        markersize = 2, color=:black, markerstrokewidth = 0, markershape =:circle, 
        label="")
    end
    # Highlight train/val region (before test starts)
    if !isnothing(train_idx) && !isempty(train_idx)
        train_range = first(train_idx) : last(train_idx)
        
        # Add vertical shaded region for train/val
        vspan!(plt, [time[first(train_range)], time[last(train_range)]];
            fillcolor=:green, alpha=0.075, label="train/val region")
    end
 
    # Highlight test region
    if !isnothing(test_idx) && !isempty(test_idx)
        test_range = first(test_idx) : last(test_idx)
        
        # Add vertical shaded region for test
        vspan!(plt, [time[first(test_range)], time[last(test_range)]];
            fillcolor=:purple, alpha=0.075, label="test region")
        
    end
    # Annotate zero crossings
    if show_zeros
        for (i, idx) in enumerate(zero_idxs)
            shift = i % 2
            scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black, markersize=3)
            annotate!(plt, time[idx], lift[idx] + 0.1 -(shift*0.2) , text(string(round(time[idx],digits = 3)), 8, :right))
        end
    end

    # display(plt)
    return plt
end

function train_force_plot(simdata::SimData; 
        train_idx=nothing, val_idx=nothing, test_idx=nothing, show_zeros=true)
   train_force_plot(simdata.force, simdata.time; 
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, show_zeros=show_zeros)
end


function velocity_flood(u::AbstractArray{T, 2}, v::AbstractArray{T, 2};title="", colorbar=false) where {T}
    plt_u = u_flood(u; colorbar=colorbar)
    plt_v = v_flood(v; colorbar=colorbar)
    plt_combined = plot(plt_u, plt_v;
        layout=(2, 1),
        size=(350, 800), 
        plot_title=title,
        titlefontsize=12,
        margin=0Plots.mm,
        framestyle=:none)
    return plt_combined, (plt_u, plt_v)
end

function u_flood(u::AbstractArray{T, 2}; colorbar=false) where {T}
    u_clims = (0.1, 2)
    flood(u;
    clims=u_clims,
    # levels=20,
    title="u-velocity",
    # xlabel="x",
    # ylabel="y",
    aspectratio=:equal,
    framestyle=:none,
    border=:none,
    colorbar=colorbar,
    titlefontsize=12,
    dpi=150,
    size=(400, 350))
end

function v_flood(v::AbstractArray{T, 2}; colorbar=false) where {T}
    v_clims = (-1, 1)
    flood(v;
    clims=v_clims,
    # levels=20,
    title="v-velocity",
    # xlabel="x",
    # ylabel="y",
    aspectratio=:equal,
    framestyle=:none,
    border=:none,
    colorbar=colorbar,
    titlefontsize=12,
    dpi=150,
    size=(400, 350))    
end

velocity_flood(u::AbstractArray{T, 3}; title="") where {T} = velocity_flood(u[:, :, 1], u[:, :, 2]; title=title)
velocity_flood(sim::AbstractSimulation; title="") = velocity_flood(remove_ghosts(sim.flow.u); title=title)



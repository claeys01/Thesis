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


function train_force_plot(forces::Vector{Vector{Float32}}, time::Vector{Float32};
        train_idx=nothing, val_idx=nothing, test_idx=nothing, show_zeros=true,
        chunk_ranges::Vector{UnitRange{Int}}=UnitRange{Int}[])
    drag = first.(forces)
    lift = last.(forces)

    ranges = isempty(chunk_ranges) ? [1:length(time)] : chunk_ranges
    xmax = max(50, ceil(maximum(time)))

    plt = plot(framestyle=:box, size=(600, 300), dpi=500,
        xlabel="\$t^*\$", ylabel="Force coefficient",
        xlims=(0, xmax), ylims=(-3, 2),
        titlefontsize=14,
        guidefontsize=12, tickfontsize=8, legendfontsize=6,
        foreground_color_axis  = :black,
        foreground_color_text  = :black,
        left_margin   = 3Plots.mm,
        right_margin  = 1Plots.mm,
        top_margin    = 1Plots.mm,
        bottom_margin = 2Plots.mm,
        legend=:topright,
        background_color_legend = RGBA(1, 1, 1, 0.7))

    for (i, rg) in enumerate(ranges)
        first_chunk = (i == 1)
        plot!(plt, time[rg], drag[rg], label=first_chunk ? L"C_{d}" : "", color=:red, lw=1)
        plot!(plt, time[rg], lift[rg], label=first_chunk ? L"C_{L}" : "", color=:blue, lw=1)
        if !first_chunk
            vline!(plt, [time[first(rg)]]; color=:gray, linestyle=:dot, alpha=0.5,
                label=(i == 2 ? "chunk start" : ""))
        end
    end

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
    # Annotate zero crossings — computed per chunk so boundaries don't create spurious crossings
    if show_zeros
        zero_idxs = Int[]
        for rg in ranges
            append!(zero_idxs, first(rg) .- 1 .+ zero_crossing(lift[rg]; direction=:rising))
        end
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
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, show_zeros=show_zeros,
        chunk_ranges=simdata.chunk_ranges)
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
    u_clims = (0, 2)
    flood(u;
    clims=u_clims,
    shift=(-2, -1.5),
    title="u-velocity",
    aspectratio=:equal,
    axis=([], false),
    framestyle=:none,
    border=:none,
    background=:white,
    cfill=:seismic,
    colorbar=colorbar,
    legend=false,
    titlefontsize=12,
    dpi=350,
    size=(400, 350))
end

function v_flood(v::AbstractArray{T, 2}; colorbar=false) where {T}
    v_clims = (-1, 1)
    flood(v;
    clims=v_clims,
    shift=(-2, -1.5),
    title="v-velocity",
    aspectratio=:equal,
    axis=([], false),
    framestyle=:none,
    border=:none,
    background=:white,
    cfill=:seismic,
    colorbar=colorbar,
    legend=false,
    titlefontsize=12,
    dpi=350,
    size=(400, 350))
end

velocity_flood(u::AbstractArray{T, 3}; title="", colorbar=false) where {T} = velocity_flood(u[:, :, 1], u[:, :, 2]; title=title, colorbar=colorbar)
velocity_flood(sim::AbstractSimulation; title="", colorbar=false) = velocity_flood(remove_ghosts(sim.flow.u); title=title, colorbar=colorbar)



function curl_plot(u::AbstractArray{T, 3}; L=32, U=1) where {T}
    ω = zeros(size(u)[1], size(u)[2])
    @inside ω[I] = WaterLily.curl(3,I,u)*L/U
    @inside ω[I] = ifelse(abs(ω[I])<0.001,0.0,ω[I])
    plt = flood(ω,shift=(-2,-1.5),clims=(-8,8), axis=([], false),  
    background=:white,
    cfill=:seismic,legend=false,border=:none,dpi=350, size=(800, 800))
    return plt
end
curl_plot(sim::AbstractSimulation) = curl_plot(sim.flow.u; L=sim.L, U=sim.U)
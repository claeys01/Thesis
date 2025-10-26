using WaterLily
using BiotSavartBCs
using Plots
using Revise 
using JLD2

includet("../custom.jl")

function circle_shedding_biot(Re=250, U=1; mem=Array)
    n = 2^7
    m = 2^7
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = (Float32(n/4), Float32(m/2)) # location of the circle relative to the height of the domain

    f = 2.5
    St = 0.2
    visc = St * Re / (f * (2*radius)^2)

    sdf(x,t) = √sum(abs2, x .- center) - radius
    sim = BiotSimulation(
        (n, m), 
        (U, 0),          # flow velocity
        2f0radius; ν=visc, # defining viscosity
        body=AutoBody(sdf), 
        mem=mem)
    perturb!(sim; noise=0.1)
    return sim
end


function zero_crossing(y; direction=:both, eps=0.0)
    @assert direction in (:both, :rising, :falling)
    n = length(y)
    idx = Int[]
    # treat tiny values as exact zeros if eps>0
    yproc = copy(y)
    if eps > 0.0
        for i in eachindex(yproc)
            if abs(yproc[i]) <= eps
                yproc[i] = zero(yproc[i])
            end
        end
    end

    for i in 1:n-1
        a, b = yproc[i], yproc[i+1]
        if a*b < 0 || a == 0 || b == 0
            dir = if a < 0 && b > 0
                :rising
            elseif a > 0 && b < 0
                :falling
            elseif a == 0 && b != 0
                b > 0 ? :rising : :falling
            elseif b == 0 && a != 0
                a > 0 ? :falling : :rising 
            else
                # flat at zero
                nothing
            end
            if dir !== nothing && (direction == :both || dir == direction)
                push!(idx, i)
            end
        end
    end
    return idx
end

function get_forces!(sim,t)
    sim_step!(sim,t,remeasure=false; verbose=true)
    force = WaterLily.pressure_force(sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end


if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()

    # sim = circle_shedding_biot(mem=Array)
    t_end = 100

    # # sim_gif!(sim;duration=t_end,clims=(-5,5),plotbody=true)

    # # sim_step!(sim, t_end; verbose=false)

    # function get_forces!(sim,t)
    #     sim_step!(sim,t,remeasure=false; verbose=true)
    #     force = WaterLily.pressure_force(sim)
    #     force./(0.5sim.L*sim.U^2) # scale the forces!
    # end

    # # Simulate through the time range and get forces
    # time = 1:0.1:t_end # time scale is sim.L/sim.U
    # forces = [get_forces!(sim,t) for t in time];
    
    time = 1:0.1:t_end # time scale is sim.L/sim.U
    # @save "data/datasets/biot_forces.jld2" forces
    @load "data/datasets/biot_forces.jld2" forces;

    #Plot it
    drag, lift = first.(forces), last.(forces)
    plt = plot(time,[drag, lift],
        labels=["drag" "lift"],
        colors=[:red, :blue],
        xlabel="tU/L",
        ylabel="Pressure force coefficients")

    zero_idxs = zero_crossing(last.(forces); direction=:rising)
    println(zero_idxs)
    
    for idx in zero_idxs
        scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black)
        annotate!(time[idx], lift[idx], (idx, 5, :left))
    end
    # annotate!([(4, 0, ("More text", 8, 45.0, :bottom, :red))])
    # display(plt)
    # savefig(plt, "figures/biot_forces.png")


    # u = sim.flow.u[:,:,1] # x velocity
    # ω = zeros(size(u));

    # @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U

    # # plt = flood(ω, clims=(-5,5), border=:none)
    # conv_diff = WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,WaterLily.quick;ν=sim.flow.ν,perdir=sim.flow.perdir)

    # # println(size(p_grad))
    # # plt = flood(p_grad[:,:,1], clims=(-0.075, 0.075))
    
    # t = sum(sim.flow.Δt)   # or sim_time(sim)*sim.L/sim.U depending on sdf time convention
    # measure!(sim, sim_time(sim))

    # # s = zeros(eltype(sim.flow.p), size(sim.flow.p))
    # # WaterLily.measure_sdf!(s, sim.body, sim_time(sim))
    # # mask = Float32.(ifelse.(s .< 0, 0f0, 1f0))

    # println(size(sim.flow.μ₀))
    # println(size(sim.flow.p), " ", size(sim.flow.σ))

    # pressure_grad = grad_p(sim.flow)
    # pressure_grad_old = grad(sim.flow.p)
    # RHS_field = 
    # # plt = flood(grad(sim.flow.p .* sim.flow.σ)
    # # plt = flood(grad(sim.flow.p)[:,:,1] .* sim.flow.μ₀[:,:,1])
    # # plt = flood(pressure_grad[:,:,1])
    # plt = flood(sim.flow.p)
    # plt = body_plot!(sim)

    # # draw grid lines for a (rows,cols) array onto an existing Plots plot
    # function overlay_grid!(plt, rows::Int, cols::Int; color=:black, lw=0.3, alpha=0.6)
    #     # cell boundaries are at 0.5, 1.5, ..., so lines align with image pixels
    #     x_min, x_max = 0.5, cols + 0.5
    #     y_min, y_max = 0.5, rows + 0.5
    #     # vertical lines
    #     for j in 0:cols
    #         x = j + 0.5
    #         plot!(plt, [x, x], [y_min, y_max], color=color, lw=lw, alpha=alpha, legend=false)
    #     end
    #     # horizontal lines
    #     for i in 0:rows
    #         y = i + 0.5
    #         plot!(plt, [x_min, x_max], [y, y], color=color, lw=lw, alpha=alpha, legend=false)
    #     end
    #     return plt
    # end

    # # overlay_grid!(plt, size(ω,1), size(ω,2); color=:gray, lw=0.4, alpha=0.5)

    # display(plt)
    # println(sim_time(sim))
    # println(mean_divergence(sim.flow.u))
    # println(mean_divergence(RHS(sim.flow)))
    # println(mean_divergence(RHS(sim.flow)))
    # savefig(plt, "biot_sim.png")
end
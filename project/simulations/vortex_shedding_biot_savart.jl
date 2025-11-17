using WaterLily
using BiotSavartBCs
using Plots
using Revise 
using JLD2

includet("../custom.jl")

function circle_shedding_biot(;Re=250, U=1, n = 2^7,m = 2^7, mem=Array)
    # n = 2^7
    # m = 2^7
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

    sim = circle_shedding_biot(mem=Array)
    t_end = 50
    sim_step!(sim, t_end; verbose=true)
    # flood(sim)
    # sim_gif!(sim; duration=10, remeasure=true, clims=(-5, 5), 
                    # ylims=(0, 130), xlims=(0,130), showaxis=false, background_color_outside=:match)
    
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
    
    # time = 1:0.1:t_end # time scale is sim.L/sim.U
    # # @save "data/datasets/biot_forces.jld2" forces
    # @load "data/datasets/biot_forces.jld2" forces;

    # #Plot it
    # drag, lift = first.(forces), last.(forces)
    # plt = plot(time,[drag, lift],
    #     labels=["drag" "lift"],
    #     colors=[:red, :blue],
    #     xlabel="tU/L",
    #     ylabel="Pressure force coefficients")

    # zero_idxs = zero_crossing(last.(forces); direction=:rising)
    # println(zero_idxs)
    
    # for idx in zero_idxs
    #     scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black)
    #     annotate!(time[idx], lift[idx], (idx, 5, :left))
    # end

end
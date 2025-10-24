using WaterLily
using BiotSavartBCs
using Plots
using Revise 

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

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()

    sim = circle_shedding_biot(mem=Array)
    t_end = 10

    # sim_gif!(sim;duration=t_end,clims=(-5,5),plotbody=true)

    sim_step!(sim, t_end; verbose=false)
    u = sim.flow.u[:,:,1] # x velocity
    ω = zeros(size(u));

    @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U

    # plt = flood(ω, clims=(-5,5), border=:none)
    conv_diff = WaterLily.conv_diff!(sim.flow.f,sim.flow.u⁰,sim.flow.σ,WaterLily.quick;ν=sim.flow.ν,perdir=sim.flow.perdir)
    p_grad = grad(sim.flow.p)
    println(size(p_grad))
    plt = flood(p_grad[:,:,1], clims=(-0.075, 0.075))
    # draw grid lines for a (rows,cols) array onto an existing Plots plot
    function overlay_grid!(plt, rows::Int, cols::Int; color=:black, lw=0.3, alpha=0.6)
        # cell boundaries are at 0.5, 1.5, ..., so lines align with image pixels
        x_min, x_max = 0.5, cols + 0.5
        y_min, y_max = 0.5, rows + 0.5
        # vertical lines
        for j in 0:cols
            x = j + 0.5
            plot!(plt, [x, x], [y_min, y_max], color=color, lw=lw, alpha=alpha, legend=false)
        end
        # horizontal lines
        for i in 0:rows
            y = i + 0.5
            plot!(plt, [x_min, x_max], [y, y], color=color, lw=lw, alpha=alpha, legend=false)
        end
        return plt
    end

    # overlay_grid!(plt, size(ω,1), size(ω,2); color=:gray, lw=0.4, alpha=0.5)

    display(plt)
    println(mean_divergence(sim.flow.u))
    println(mean_divergence(RHS(sim.flow)))
    # savefig(plt, "biot_sim.png")
end
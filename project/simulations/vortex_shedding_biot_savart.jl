using WaterLily
import WaterLily: S
using BiotSavartBCs
using Plots
using Revise 
using JLD2

includet("../custom.jl")

function circle_shedding_biot(;Re=250, U=1, n = 2^7,m = 2^7, mem=Array)
    
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = (Float32(n/4), Float32(m/2)) # location of the circle relative to the height of the domain


    D    = 2f0 * radius
    visc = U * D / Re        # <-- key change: ν ∝ 1/Re

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




function get_forces!(sim,t)
    sim_step!(sim,t,remeasure=false; verbose=true)
    force = WaterLily.pressure_force(sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end




if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()

    # n=2^8
    # sim = circle_shedding_biot(mem=Array, Re=2500, n=n, m=n)
    # t_end = 50
    # sim_gif!(sim; duration=t_end, remeasure=true, clims=(-5, 5), 
                    # ylims=(0, n+2), xlims=(0,n+2), showaxis=false, background_color_outside=:match)
    # u = sim.flow.u
    # u = remove_ghosts(u)[:,:,:,end]
    # strain = strain_field(u)
    # @show strain[1,1,:,:]
    # ε = kinetic_energy_diffusion(sim.flow.u; ν=sim.flow.ν)
    # @show size(ε)
    # # CFL(a::Flow;Δt_max=0.1) = 0.1

    # # sim_step!(sim, t_end; verbose=true)
    # # sim_gif!(sim; duration=t_end, remeasure=true, clims=(-5, 5), 
    #                 # ylims=(0, 130), xlims=(0,130), showaxis=false, background_color_outside=:match)
    
    # function get_forces!(sim,t)
    #     sim_step!(sim,t,remeasure=false, verbose=true)
    #     force = WaterLily.pressure_force(sim)
    #     force./(0.5sim.L*sim.U^2) # scale the forces!
    # end

    # # Simulate through the time range and get forces
    # time = 1:0.1:t_end # time scale is sim.L/sim.U
    # forces = [get_forces!(sim,t) for t in time];
    # println(size(forces))
    # # Plot it
    # plot(time,[first.(forces) last.(forces)],
    #     labels=["drag" "lift"],
    #     xlabel="tU/L",
    #     ylabel="Pressure force coefficients")

end
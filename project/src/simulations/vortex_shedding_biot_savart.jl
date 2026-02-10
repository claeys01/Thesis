function circle_shedding_biot(;Re=250, U=1, n = 2^7,m = 2^7, mem=Array, perturb=true)
    
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
    perturb && perturb!(sim; noise=0.1)
    return sim
end


function get_forces!(sim,t)
    sim_step!(sim,t,remeasure=false; verbose=true)
    force = WaterLily.pressure_force(sim)
    force./(0.5sim.L*sim.U^2) # scale the forces!
end




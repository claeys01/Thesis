using WaterLily
using BiotSavartBCs


function circle_shedding_biot(Re=250, U=1; mem=Array)
    n = 3*2^7
    m = 2^7
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = Float32(m / 2) # location of the circle relative to the height of the domain

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

sim = circle_shedding_biot(mem=Array)
t_end = 50

sim_step!(sim, t_end; verbose=true)
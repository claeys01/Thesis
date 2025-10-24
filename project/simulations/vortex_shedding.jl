using WaterLily
# using CUDA

# Define simulation size, geometry dimensions, viscosity
function circle_shedding(Re=250, U=1; mem=Array)
    n = 3*2^7
    m = 2^7
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = Float32(m / 2)  # location of the circle relative to the height of the domain

    f = 2.5
    St = 0.2
    visc = St * Re / (f * (2*radius)^2)

    sdf(x,t) = √sum(abs2, x .- center) - radius
    sim = Simulation(
            (n, m),          # domain size
            (U, 0),          # flow velocity
            2f0radius; ν=visc, # defining viscosity
            body=AutoBody(sdf),
            mem
        )
    perturb!(sim; noise=0.1)
    return sim
end

# sim = circle_shedding()
# sim_step!(sim, 2; verbose=true)
# gif = sim_gif!(sim;duration=50,clims=(-5,5),plotbody=true)

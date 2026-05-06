using Thesis
using WaterLily
using Plots
using WaterLily: flood, body_plot!

n = 2^8
sim = circle_shedding_biot(;n=n, m=n)

u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
sim.flow.u .= u₀


next_delta = 0.5
next_plot = copy(sim_time(sim)) + next_delta
counter = 1
t_end=10
while sim_time(sim) < t_end
    sim_step!(sim)
    if next_plot < sim_time(sim)
        sim_info(sim)
        @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])

        plt = flood(sim.flow.σ,shift=(-2,-1.5),clims=(-8,8), axis=([], false),  
        background=:gray,
        cfill=:seismic,legend=false,border=:none,dpi=350, size=(800, 800))
        bod = body_plot!(sim)
        timestep = counter * next_delta
        display(plt)
        savefig(plt, "figs/biot_shedding_plots/shedding_t$timestep.png")
        next_plot += next_delta
        counter +=1
    end
end

# function run_oscillating_flow(n=2^8, stop=20)
#     sim = circle_shedding_biot(;n=n,m=n)
#     sim_step!(sim,0.1)

#     @time @gif for tᵢ in range(0.,stop;step=0.2)
#         println("tU/L=",round(tᵢ,digits=4))
#         sim_step!(sim,tᵢ)
#         @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#         @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
#         # It's important to have `|>Array` during GPU simulation as `flood` only accept CPU Array input
#         flood(sim.flow.σ|>Array,shift=(-2,-1.5),clims=(-8,8), axis=([], false),
#               cfill=:seismic,legend=false,border=:none,size=(800,800), dpi=350)
#         body_plot!(sim)
#     end
# end

# run_oscillating_flow()
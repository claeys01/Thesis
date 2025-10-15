using WaterLily
using JLD2
using Statistics
using Revise
using Todo


includet("../simulations/vortex_shedding.jl")
includet("../custom.jl")

sim = circle_shedding(mem=Array)

sim_step!(sim, 2; verbose=true)
sim_rhs = RHS(sim.flow)

function divergence_field(u) 
    H,W,_ = size(u)
    σ = zeros(eltype(u), H, W)
    @inside σ[I] = WaterLily.div(I, u)
    println(size(u), size(σ))
    return σ
end



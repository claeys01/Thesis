using WaterLily

include("../basic_sim.jl")


function custom_step!(sim::WaterLily.AbstractSimulation;remeasure=true,λ=WaterLily.quick,udf=nothing,kwargs...)
    remeasure && measure!(sim)
    custom_mom_step!(sim.flow, sim.pois; λ, udf, kwargs...)
end


function custom_mom_step!(flow::WaterLily.Flow{N}, pois::WaterLily.AbstractPoisson; λ=WaterLily.quick, udf=nothing, kwargs...) where N
    """"i want to get the RHS of the momentum equation, should be conv_diff - grad_p"""

    flow.u⁰ .= flow.u; WaterLily.scale_u!(flow,0); t₁ = sum(flow.Δt); t₀ = t₁-flow.Δt[end]

    WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir)
end

sim = basic(2^4, 2^4)

custom_step!(sim)

# println(sim)
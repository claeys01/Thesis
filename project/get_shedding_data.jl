using WaterLily
import WaterLily: ∂, @loop
using Revise
using JLD2

includet("simulations/vortex_shedding.jl")
includet("custom.jl")


sim_shedding = circle_shedding(mem=Array)
t_end = 100.0

function data_run(sim::AbstractSimulation, time_max; sample_instance=50, verbose=false)
    data = Dict("RHS" => [], "time" => [], "Δt" => [])
    sample_counter = 0
    while sim_time(sim) < time_max
        sim_step!(sim)
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            sample_counter += 1
            print("Sampling RHS - ")
            push!(data["RHS"], RHS(sim.flow))
            push!(data["time"], Float32(round(sim_time(sim),digits=4)))
            push!(data["Δt"], Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    println("Sampled ", sample_counter," RHS")
    return data
end

RHS_data = data_run(sim_shedding, t_end; verbose=true)

@save "data/RHS_shedding_data_arr_temp.jld2" RHS_data
using WaterLily
import WaterLily: ∂, @loop
using Revise
using JLD2

includet("simulations/vortex_shedding.jl")
includet("simulations/vortex_shedding_biot_savart.jl")
includet("custom.jl")


sim_shedding = circle_shedding_biot(mem=Array)
t_end = 100.0

function data_run(sim::AbstractSimulation, time_max; sample_instance=50, verbose=false)
    data = Dict("RHS" => [], "time" => [], "Δt" => [], "flow" => [])
    sample_counter = 0
    while sim_time(sim) < time_max
        sim_step!(sim)
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            sample_counter += 1
            print("Sampling RHS - ")
            push!(data["flow"], sim.flow)
            push!(data["RHS"], RHS(sim.flow))
            push!(data["time"], Float32(round(sim_time(sim),digits=4)))
            push!(data["Δt"], Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    println("Sampled ", sample_counter," RHS")
    return data
end

RHS_data = data_run(sim_shedding, t_end; verbose=true)

@save "data/flow_RHS_biot_data_arr_new.jld2" RHS_data
using WaterLily
import WaterLily: ∂, @loop
using Revise
using JLD2

# includet("simulations/vortex_shedding.jl")
includet("simulations/vortex_shedding_biot_savart.jl")
includet("custom.jl")


sim_shedding = circle_shedding_biot(mem=Array)
t_end = 100.0

function data_run(sim::AbstractSimulation, time_max, save_path; sample_instance=50, verbose=false, sample_single_period=false, single_period_direction=:rising)
    data = Dict{String,Any}(
        "RHS"   => Any[],
        "time"  => Any[],
        "Δt"    => Any[],
        "flow"  => Any[],
        "force" => Any[],
    )
    sample_counter = 0
    while sim_time(sim) < time_max
        sim_step!(sim)
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            force = WaterLily.pressure_force(sim)
            force = force./(0.5sim.L*sim.U^2) # scale the forces!
            sample_counter += 1
            print("Sampling RHS - ")
            push!(data["force"], force)
            push!(data["flow"], sim.flow)
            push!(data["RHS"], RHS(sim.flow))
            push!(data["time"], Float32(round(sim_time(sim),digits=4)))
            push!(data["Δt"], Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    println("Sampled ", sample_counter," RHS")

    @info "Saved $(sample_counter) snapshots to $(save_path)"
    if sample_single_period
        split_path = split(save_path, ".")
        single_period_save = split_path[1] * "_period." * split_path[2]
        zero_idxs = zero_crossing(last.(data["force"]); direction=single_period_direction)
        if length(zero_idxs) >= 2
            mid = length(zero_idxs) ÷ 2
            # ensure valid mid index
            if mid < length(zero_idxs)
                single_period = zero_idxs[mid] : zero_idxs[mid+1]
                period_data = Dict()
                for key in keys(data)
                    period_data[key] = data[key][single_period]
                end
                # save the single-period dictionary and indices
                @save single_period_save RHS_data = period_data
                @info "Saved data from a single period to $(single_period_save)"

                # also attach to returned data for convenience
                data["single_period"] = period_data
            else
                @warn "Unable to determine single period: not enough zero crossings in the middle"
            end
        else
            @warn "Not enough zero crossings to extract a single period"
        end
    end

    @save save_path RHS_data = data

    return data
end

# RHS_data = data_run(sim_shedding, t_end, "data/RHS_biot_data_arr_force.jld2"; verbose=true, sample_single_period=true)

@load "data/RHS_biot_data_arr_force.jld2" RHS_data

forces = RHS_data["force"]
time = RHS_data["time"]
zero_idxs = zero_crossing(last.(forces); direction=:rising)
drag, lift = first.(forces), last.(forces)

plt = plot(time,[drag, lift],
    labels=["drag" "lift"],
    colors=[:red, :blue],
    xlabel="tU/L",
    ylabel="Pressure force coefficients")
zero_idxs = zero_crossing(last.(forces); direction=:rising)

for idx in zero_idxs
    scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black)
    annotate!(time[idx], lift[idx], (idx, 5, :left))
end

mid = length(zero_idxs) ÷ 2
single_period = zero_idxs[mid] : zero_idxs[mid+1]
period_data = RHS_data["single_period"]
plot!(period_data["time"], last.(period_data["force"]); 
    linestyle =:dash, lw=2, label="sampled period", color=:green)


display(plt)
savefig(plt, "figs/period_sample.png")


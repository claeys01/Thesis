using WaterLily
import WaterLily: ∂, @loop
using Revise
using JLD2

# includet("simulations/vortex_shedding.jl")
includet("../simulations/vortex_shedding_biot_savart.jl")
includet("../custom.jl")

n=2^8
sim_shedding = circle_shedding_biot(mem=Array, Re=2500, n=n, m=n)
t_end = 100.0

function data_run(sim::AbstractSimulation, time_max, save_path; sample_instance=50, verbose=false, sample_single_period=false, single_period_direction=:rising)
    data = Dict{String,Any}(
        "time"  => Float32[],
        "Δt"    => Float32[],
        "u"     => AbstractArray[],
        "μ₀"    => AbstractArray[],
        "force" => Any[],
        "ε"     => Float32[],
    )
    sample_counter = 0
    root, ext = splitext(save_path)
    period_path = string(root, "_period", ext)
    full_path   = string(root, "_full",   ext)
    while sim_time(sim) < time_max
        sim_step!(sim)
        
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            force = WaterLily.pressure_force(sim)
            force = force./(0.5sim.L*sim.U^2) # scale the forces!
            sample_counter += 1
            print("Sampling Data - ")
            push!(data["force"], force)
            # push!(data["flow"], sim.flow)                     # maybe keep reference if intended
            push!(data["u"], copy(sim.flow.u))               # make a snapshot copy
            push!(data["μ₀"], copy(sim.flow.μ₀))             # make a snapshot copy
            # push!(data["RHS"], copy(RHS(sim.flow)))         # ensure RHS is a separate array
            push!(data["ε"], mean(kinetic_energy_diffusion(copy(sim.flow.u); ν=copy(sim.flow.ν))))
            push!(data["time"], Float32(round(sim_time(sim),digits=4)))
            push!(data["Δt"], Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    println("Sampled ", sample_counter," Snapshots")
    for k in ["RHS", "u", "μ₀"]
        if haskey(data, k) && !isempty(data[k]) && isa(first(data[k]), AbstractArray)
            sample0 = first(data[k])
            if ndims(sample0) >= 2
                dims = 4
                try
                    data[k] = cat(data[k]...; dims=dims)
                    @info "Stacked samples for $(k) into size $(size(data[k]))"
                catch e
                    @warn "Failed to stack samples for $(k): $e"
                end
            end
        end
    end
    
    @info "Saved $(sample_counter) snapshots to $(save_path)"
    if sample_single_period
        zero_idxs = zero_crossing(last.(data["force"]); direction=single_period_direction)
        if length(zero_idxs) >= 2
            mid = length(zero_idxs) ÷ 2
            # ensure valid mid index
            if mid < length(zero_idxs)
                single_period = zero_idxs[mid] : zero_idxs[mid+1]
                println(single_period)
                period_data = Dict()
                period_data["single_period"] = single_period
                for key in keys(data)
                    if key in ["RHS", "u", "μ₀"]
                        period_data[key] = data[key][:,:,:,single_period]
                    else
                        period_data[key] = data[key][single_period]
                    end
                end
                # save the single-period dictionary and indices
                @save period_path data = period_data
                @info "Saved data from a single period to $(period_path)"

                # also attach to returned data for convenience
                data["single_period"] = period_data
            else
                @warn "Unable to determine single period: not enough zero crossings in the middle"
            end
        else
            @warn "Not enough zero crossings to extract a single period"
        end
    end

    @save full_path data

    return data
end

save_dir = "data/datasets/2e8/RE2500/"
data_path = joinpath(save_dir, "U_128.jld2")

data = data_run(sim_shedding, t_end, "data/datasets/2e8/RE2500/U_128.jld2"; verbose=true, sample_single_period=true)

# @load data_path data


forces = data["force"]
time = data["time"]
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
# println(zero_idxs)
single_period = zero_idxs[mid] : zero_idxs[mid+1]
period_data = data["single_period"]
plot!(period_data["time"], last.(period_data["force"]); 
    linestyle =:dash, lw=2, label="sampled period", color=:green)


display(plt)
fig_path= joinpath(save_dir,"period_sample.png")
savefig(plt, fig_path)


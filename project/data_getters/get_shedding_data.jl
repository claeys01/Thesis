using WaterLily
import WaterLily: ∂, @loop, pressure_force
using Revise
using JLD2

# Ensure parent directory exists for any file path
function ensure_parent_dir(path::AbstractString)
    dir = dirname(path)
    if !ispath(dir)
        mkpath(dir)
    end
    return dir
end

# includet("simulations/vortex_shedding.jl")
includet("../simulations/vortex_shedding_biot_savart.jl")
includet("../custom.jl")
includet("../utils/SimDataTypes.jl")

using .SimDataTypes

function reorder_center_out(arr)
    n = length(arr)
    mid = (firstindex(arr) + lastindex(arr)) ÷ 2
    out = eltype(arr)[]

    push!(out, arr[mid])

    for k in 1:max(mid - firstindex(arr), lastindex(arr) - mid)
        left  = mid - k
        right = mid + k

        if left >= firstindex(arr)
            push!(out, arr[left])
        end
        if right <= lastindex(arr)
            push!(out, arr[right])
        end
    end

    return out
end



function data_run(sim::AbstractSimulation, time_max, save_path; sample_instance=50, verbose=false, sample_single_period=false, single_period_direction=:rising)
  
    # Temporary storage while sampling
    time   = Float32[]
    Δt     = Float32[]
    u_list = Vector{Array{Float32,3}}()          # assuming u is 3D per snapshot
    μ₀_list = Vector{Array{Float32,3}}()         # adjust if needed
    force  = Vector{Vector{Float32}}()
    ε      = Float32[]

    sample_counter = 0
    root, ext = splitext(save_path)
    period_path = string(root, "_period", ext)
    full_path   = string(root, "_full",   ext)

    # Create parent dirs for both outputs
    ensure_parent_dir(period_path)
    ensure_parent_dir(full_path)
    
    while sim_time(sim) < time_max
        sim_step!(sim)
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            raw_force = pressure_force(sim)
            scaled_force = Float32.(raw_force./(0.5sim.L*sim.U^2)) # scale the forces!
            sample_counter += 1
            print("Sampling Data - ")
            push!(force, scaled_force)
            push!(u_list, copy(sim.flow.u))               # make a snapshot copy
            push!(μ₀_list, copy(sim.flow.μ₀))             # make a snapshot copy
            push!(ε, mean(kinetic_energy_dissipation(copy(sim.flow.u); ν=copy(sim.flow.ν))))
            push!(time, Float32(round(sim_time(sim),digits=4)))
            push!(Δt, Float32(round(sim.flow.Δt[end], digits=3)))
        end
    end
    println("Sampled ", sample_counter," Snapshots")

    # Stack snapshots into dense arrays
    # u_list :: Vector{Array{T,3}} -> u :: Array{T,4}
    u = cat(u_list...; dims = 4)  # (nx, ny, nchan, Nsnap)

    # μ₀_list :: Vector{Array{T,1}} -> μ₀ :: Array{T,4} (nx, ny, nchan, Nsnap)
    μ₀ = cat(μ₀_list...; dims = 4)

    # zero crossings based on lift (second component)
    lift = last.(force)
    zero_idxs = zero_crossing(lift; direction = single_period_direction)

    if isempty(zero_idxs)
        @warn "No zero crossings found; period information will be empty"
    end

    time .-= time[1]

    @info "Saved $(sample_counter) snapshots to $(save_path)"
   if sample_single_period && length(zero_idxs) >= 2
        mid = length(zero_idxs) ÷ 2
        if mid < length(zero_idxs)
            single_period_idx = zero_idxs[mid] : zero_idxs[mid + 1]
            @info "Single period indices: $single_period_idx"

            # Optional: save just the single-period subset as a separate file
            period_data = (
                time   = time[single_period_idx],
                Δt     = Δt[single_period_idx],
                u      = u[:, :, :, single_period_idx],
                μ₀     = μ₀[:, :, :, single_period_idx],
                force  = force[single_period_idx],
                ε      = ε[single_period_idx],
                single_period_idx = single_period_idx,
            )
            @save period_path period_data
            @info "Saved data from a single period to $(period_path)"
        else
            @warn "Unable to determine single period: not enough zero crossings in the middle"
        end
    end

    # Build period_ranges + reordered_ranges
    period_ranges = length(zero_idxs) >= 2 ?
        [(zero_idxs[i - 1] + 1):zero_idxs[i] for i in 2:length(zero_idxs)] :
        UnitRange{Int}[]

    reordered_ranges = reorder_center_out(period_ranges)
    simdata = SimData(time, Δt, u, μ₀, force, ε,
                      period_ranges, reordered_ranges, single_period_idx)

    @save full_path simdata
    @info "Saved $(sample_counter) snapshots to $(full_path)"

    return simdata
end

# # save_dir = "data/datasets/RE2500/2e8/"
# save_dir = "data/datasets/RE250/"
# name = "U_128.jl"               # change depending on grid size
# data_path = joinpath(save_dir, name)

# root, ext = splitext(data_path)
# period_path = string(root, "_period", ext)
# full_path   = string(root, "_full",   ext)


# simdata = data_run(sim_shedding, t_end, data_path; verbose=true, sample_single_period=true)

# @load full_path simdata


function plot_sampled_period(simdata::Any, period_path::AbstractString, save_dir::AbstractString)
    forces = simdata.force
    time = simdata.time
    drag = first.(forces)
    lift = last.(forces)

    zero_idxs = zero_crossing(lift; direction=:rising)

    plt = plot(time, [drag, lift],
        labels=["drag" "lift"],
        colors=[:red, :blue],
        xlabel="tU/L",
        ylabel="Pressure force coefficients")

    for idx in zero_idxs
        scatter!(plt, [time[idx]], [lift[idx]]; label=false, color=:black)
        annotate!(plt, time[idx], lift[idx] + 0.1, text(string(idx), 8, :left))
    end

    mid = length(zero_idxs) ÷ 2
    single_period = (length(zero_idxs) >= 2 && mid < length(zero_idxs)) ? (zero_idxs[mid] : zero_idxs[mid+1]) : nothing

    if isfile(period_path)
        @load period_path period_data
        if period_data !== nothing
            plot!(plt, period_data.time, last.(period_data.force);
                linestyle=:dash, lw=2, label="sampled period", color=:green)
            plot!(plt, period_data.time, first.(period_data.force);
                linestyle=:dash, lw=2, label="", color=:green)
        else
            @warn "period file did not contain `period_data`: $period_path"
        end
    else
        @warn "period file not found: $period_path"
    end

    display(plt)
    fig_path = joinpath(save_dir, "period_sample.png")
    savefig(plt, fig_path)

    return (plt = plt, fig_path = fig_path, zero_idxs = zero_idxs, single_period = single_period)
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    n = 2^8
    Re = 2500

    # Build save_dir and ensure it exists
    save_dir = string("data/datasets/RE",Re, "/2e", Int(log2(n)), "/" )
    if !ispath(save_dir)
        mkpath(save_dir)
    end

    name = string("U_", n, ".jld2")            
    data_path = joinpath(save_dir, name)

    root, ext = splitext(data_path)
    period_path = string(root, "_period", ext)
    full_path   = string(root, "_full",   ext)

    @show save_dir, name, data_path
    @show period_path, full_path

    sim_shedding = circle_shedding_biot(mem=Array, Re=Re, n=n, m=n)
    t_end = 100.0

    simdata = data_run(sim_shedding, t_end, data_path; verbose=true, sample_single_period=true)

    plot_sampled_period(simdata, period_path, save_dir)
end




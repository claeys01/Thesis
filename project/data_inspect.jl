using JLD2
using WaterLily
using WaterLily: @inside
using Statistics
using Plots

# load helpers from project
includet("custom.jl")   # provides grad, RHS, get_random_snapshots, downsample_RHS_data!, remove_ghosts



"""
    inspect_RHS_data(path_or_RHS; kwargs...)

Inspect RHS dataset (either the Dict loaded from JLD2 or a filename). Selects
random snapshot(s), prints statistics and plots each snapshot.

Keyword args:
- n=1         : number of random snapshots to plot
- seed=42
- tmin,tmax   : time window passed to get_random_snapshots
- downsample  : downsample argument passed to get_random_snapshots
- clip_bc=true: remove ghost cells before analysis (recommended)
- verbose=true
"""
function inspect_RHS_data(path_or_RHS; n::Int=1, seed::Int=42, tmin=-1, tmax=-1, downsample=-1, clip_bc=true, verbose=true)
    # delegate loading/downsampling/selection to helper in custom.jl
    random_RHS, random_flow, inds = get_random_snapshots(path_or_RHS; n=n, seed=seed, tmin=tmin, tmax=tmax, downsample=downsample, clip_bc=clip_bc, verbose=verbose)
    # snapshots shaped (H, W, C, n)
    for k in 1:n
        RHS = random_RHS[:,:,:,k]
        u = random_flow[k].u
        println("Random Snapshot #$k  (original index = $(inds[k]))  size=$(size(RHS))  eltype=$(eltype(RHS))")
        plots = []
        comps = ["x-comp", "y-comp"]
        # collect stats
        flow_stats = Dict{String,NamedTuple}()
        rhs_stats  = Dict{String,NamedTuple}()
        for comp in 1:2
            u_comp = u[:,:,comp]; RHS_comp = RHS[:,:,comp]
            flow_stats[comps[comp]] = (mean=mean(u_comp),  std=std(u_comp))
            rhs_stats[comps[comp]]  = (mean=mean(RHS_comp), std=std(RHS_comp))
            up = flood(u_comp, border=:none; title="$(comps[comp]) of velocity field", titlefontsize=10)
            rhsp = flood(RHS_comp, border=:none; title="$(comps[comp]) of RHS field", titlefontsize=10)
            push!(plots, up)
            push!(plots, rhsp)
        end

        @info "Snapshot $(k) (orig index $(inds[k])) stats" flow=flow_stats rhs=rhs_stats

        @info "Mean divergence:" flow = mean_divergence(u) RHS = mean_divergence(RHS)

        u_mag = dropdims(sqrt.(sum(u .^ 2, dims=3)), dims=3) 
        RHS_mag = dropdims(sqrt.(sum(RHS .^ 2, dims=3)), dims=3)

        u_magp = flood(u_mag, border=:none; title="magnitude of velocity field", titlefontsize=10); push!(plots, u_magp)
        RHS_magp = flood(RHS_mag, border=:none; title="magnitude of RHS field", titlefontsize=10); push!(plots, RHS_magp)


        p = plot(plots..., layout=(3,2), size=(600,750))
        display(p)
    end
    # return snapshots, inds
end


function inspect_mu(path; n::Int=1, seed::Int=42, tmin=-1, tmax=-1, downsample=-1, clip_bc=false, verbose=true)
    random_RHS, random_flow, inds = get_random_snapshots(path; n=n, seed=seed, tmin=tmin, tmax=tmax, downsample=downsample, clip_bc=clip_bc, verbose=verbose)
    for k in 1:n
        flow = random_flow[k]

        plt = flood(flow.μ₀[:,:,2])
        display(plt)
    end
end



# Convenience CLI-like behaviour when file is run interactively
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    # try to load default file if present
    default_file = "data/datasets/RHS_biot_data_arr_force_period.jld2"
    # println("Inspecting default RHS file: $default_file")
    inspect_mu(default_file)

    # inspect_RHS_data(default_file; n=1, seed=42, clip_bc=false, verbose=true)
    # nothing

    # new_file = "data/RHS_biot_data_arr_new2.jld2"
    # println("Inspecting default RHS file: $new_file")
    # inspect_RHS_data(new_file; n=1, seed=42, clip_bc=false, verbose=true)

end
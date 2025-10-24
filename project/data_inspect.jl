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
    snapshots, inds = get_random_snapshots(path_or_RHS; n=n, seed=seed, tmin=tmin, tmax=tmax, downsample=downsample, clip_bc=clip_bc, verbose=verbose)
    # snapshots shaped (H, W, C, n)
    ns = size(snapshots, 4)
    for k in 1:ns
        s = snapshots[:,:,:,k]
        idx = inds[k]
        println("Snapshot #$k  (original index = $(idx))  size=$(size(s))  eltype=$(eltype(s))")

        # component fields (s[:,:,i] already returns a 2D array)
        u = s[:,:,1]
        v = s[:,:,2]

        u_mean, u_std = mean(u), std(u)
        v_mean, v_std = mean(v), std(v)
        mag = dropdims(sqrt.(sum(s .^ 2, dims=3)), dims=3)   # remove singleton component axis
        mag_mean, mag_std = mean(mag), std(mag)

        # divergence
        div_mean = try
            mean_divergence(s)
        catch e
            @warn "mean_divergence failed: $e"
            NaN
        end

        println("u: mean=$(round(u_mean, sigdigits=6)), std=$(round(u_std, sigdigits=6))")
        println("v: mean=$(round(v_mean, sigdigits=6)), std=$(round(v_std, sigdigits=6))")
        println("|u|: mean=$(round(mag_mean, sigdigits=6)), std=$(round(mag_std, sigdigits=6))")
        println("mean(divergence) = $(round(div_mean, sigdigits=6))")

        # plotting
        px = flood(u, border=:none, clims=(u_mean-u_std, u_mean+u_std))
        py = flood(v, border=:none, clims=(v_mean-v_std, v_mean+v_std))
        pmag = flood(mag, border=:none, clims=(mag_mean-mag_std, mag_mean+mag_std))

        p = plot(px, py, pmag, layout=(3,1), size=(500,750), title=["u" "v" "|u|"])
        display(p)
    end
    return snapshots, inds
end

# Convenience CLI-like behaviour when file is run interactively
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    # try to load default file if present
    default_file = "data/RHS_biot_data_arr.jld2"
    println("Inspecting default RHS file: $default_file")
    inspect_RHS_data(default_file; n=1, seed=42, clip_bc=false, verbose=true)

    new_file = "data/RHS_biot_data_arr_new2.jld2"
    println("Inspecting default RHS file: $new_file")
    inspect_RHS_data(new_file; n=1, seed=42, clip_bc=false, verbose=true)


end
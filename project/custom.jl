using WaterLily
import WaterLily: ∂, @loop, @inside
using JLD2

function grad(field::AbstractArray)
    T = eltype(field)
    sz = size(field)
    N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function RHS(flow::Flow{N};λ=WaterLily.quick,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad(flow.p)
    return RHS
end

function remove_ghosts(snapshot::AbstractArray)
        return snapshot[2:end-1, 2:end-1, :]
    return out
end

function downsample_RHS_data!(RHS_data; tmin=-1, tmax=-1, n_samples=-1, clip_bc=false, verbose=true)
    if n_samples == -1
        n_samples = length(RHS_data["time"])
    end
    if tmin == -1
        tmin = RHS_data["time"][1]
    end
    if tmax == -1
        tmax = RHS_data["time"][end]
    end 
    # Select indices corresponding to time tmin to tmax
    time_indices = findall(t -> t ≥ tmin && t ≤ tmax, RHS_data["time"])
    selected_indices = time_indices

    # Downsample to n_samples entries
    downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
    final_indices = selected_indices[downsampled_indices]

    # Downsample all relevant entries in RHS_data
    RHS_data["time"] = RHS_data["time"][final_indices]
    RHS_data["Δt"] = RHS_data["Δt"][final_indices]
    RHS_data["RHS"] = RHS_data["RHS"][final_indices]

    if clip_bc
        # Exclude outer cells (boundary) from each RHS entry
        for (i, RHS) in enumerate(RHS_data["RHS"])
            RHS_data["RHS"][i] = remove_ghosts(RHS)
        end
    end
    if verbose
        @info "Downsampled to $(length(RHS_data["time"])) time steps."
        @info "Input data size: $(size(RHS_data["RHS"][1]))"
    end

end

function get_random_snapshots(path_or_RHS; n::Int=5, seed::Int=42,
                              tmin=-1, tmax=-1, downsample=-1, clip_bc=true, verbose=false)
    # load or accept RHS_data dict
    RHS_data = if isa(path_or_RHS, AbstractString)
        @load path_or_RHS RHS_data
        RHS_data
    elseif isa(path_or_RHS, Dict)
        deepcopy(path_or_RHS)   # avoid mutating caller's dict
    else
        throw(ArgumentError("path_or_RHS must be a filename or a Dict as loaded from JLD2"))
    end

    # downsample / clip in-place on our copy
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=downsample, clip_bc=clip_bc, verbose=verbose)

    # build 4-D array (H,W,C,N)
    X = cat(RHS_data["RHS"]...; dims=4)
    X = Float32.(X)
    nsnaps = size(X, 4)

    if nsnaps == 0
        throw(ArgumentError("No snapshots available in RHS_data"))
    end

    if n > nsnaps
        @warn "Requested $n snapshots but only $nsnaps available — returning all"
        n = nsnaps
    end

    rng = MersenneTwister(seed)
    inds = randperm(rng, nsnaps)[1:n]
    snapshots = X[:, :, :, inds]   # (H,W,C,n)

    return snapshots, collect(inds)
end
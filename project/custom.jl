using WaterLily
import WaterLily: ∂, @loop, @inside
using JLD2
using Random
using Statistics

function grad(field::AbstractArray)
    T = eltype(field); sz = size(field); N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function grad_p(flow::Flow)
    return grad(flow.p) .* flow.μ₀
end

function RHS(flow::Flow{N};λ=WaterLily.quick,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad_p(flow)
    return RHS
end

function remove_ghosts(snapshot::AbstractArray)
    return snapshot[2:end-1, 2:end-1, :]
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

    if haskey(RHS_data, "flow")
        RHS_data["flow"] = RHS_data["flow"][final_indices]
    end

    if haskey(RHS_data, "μ₀")
        RHS_data["μ₀"] = RHS_data["μ₀"][final_indices]
    end

    if clip_bc
        # Exclude outer cells (boundary) from each RHS entry
        for (i, RHS) in enumerate(RHS_data["RHS"])
            RHS_data["RHS"][i] = remove_ghosts(RHS)
        end
        if haskey(RHS_data, "μ₀")
            for (j, mask) in enumerate(RHS_data["μ₀"])
                RHS_data["μ₀"][j] = remove_ghosts(mask)
            end
        end
    end
    if verbose
        @info "Downsampled to $(length(RHS_data["time"])) time steps."
        @info "Input data size: $(size(RHS_data["RHS"][1]))"
    end

end

"""
    mean_divergence(a)

Compute the mean divergence of a staggered vector field `a` whose components
are stored on the last axis (e.g. Nx×Ny×2). Returns the mean over interior cells.
"""
function mean_divergence(a::AbstractArray)
    nd = ndims(a)
    spatial_dim = nd - 1
    if spatial_dim <= 0
        throw(ArgumentError("mean_divergence: expected array with component axis; got ndims=$nd"))
    end
    ncomp = size(a, nd)
    if ncomp < spatial_dim
        throw(ArgumentError("mean_divergence: expected last axis to contain ≥ $spatial_dim components, got $ncomp"))
    end

    spat_sizes = Tuple(size(a)[1:spatial_dim])
    init = zeros(eltype(a), spat_sizes...)   # scalar field to hold divergence

    # compute divergence on interior cells only (consistent with library's conventions)
    @inside init[I] = WaterLily.div(I, a)

    # If there are no interior cells (small arrays) fall back to computing over all valid indices
    if sum(!iszero, init) == 0 && length(init) > 0
        # compute directly over CartesianIndices(init) but skipping boundaries
        for I in CartesianIndices(init)
            try
                init[I] = WaterLily.div(I, a)
            catch
                # ignore indices where div cannot be evaluated
            end
        end
    end

    return mean(init)
end

function get_random_snapshots(path_or_RHS; n::Int=5, seed::Int=42,
                              tmin=-1, tmax=-1, downsample=-1, clip_bc=true, verbose=false)
    # load or accept RHS_data dict
    RHS_data = if isa(path_or_RHS, AbstractString)
        @load path_or_RHS RHS_data
        # ensure we have a Dict with Any values so we can replace entries with Arrays
        Dict{String,Any}(RHS_data)
    elseif isa(path_or_RHS, Dict)
        # make a writable copy with Any-typed values to allow in-place replacement
        Dict{String,Any}(path_or_RHS)
    else
        throw(ArgumentError("path_or_RHS must be a filename or a Dict as loaded from JLD2"))
    end

    # downsample / clip in-place on our copy
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=downsample, clip_bc=clip_bc, verbose=verbose)

    # build 4-D array (H,W,C,N)
    get_4d!(RHS_data)

    nsnaps = size(RHS_data["RHS"], 4)

    if nsnaps == 0
        throw(ArgumentError("No snapshots available in RHS_data"))
    end

    if n > nsnaps
        @warn "Requested $n snapshots but only $nsnaps available — returning all"
        n = nsnaps
    end

    rng = MersenneTwister(seed)
    inds = randperm(rng, nsnaps)[1:n]
    random_RHS = RHS_data["RHS"][:, :, :, inds]   # (H,W,C,n)
    random_flow = try
        RHS_data["flow"][inds]
    catch e
        @warn "flow key not present in this database: $e"
        zeros(size(random_RHS))
    end
    return random_RHS, random_flow, collect(inds)
end

function get_4d!(RHS_data::Dict)
    # build a 4-D array from whatever is stored under "RHS" and replace it in-place
    arr = RHS_data["RHS"]
    new = if isa(arr, AbstractArray) && ndims(arr) == 4
        Float32.(arr)
    else
        Float32.(cat(arr...; dims=4))
    end
    RHS_data["RHS"] = new
    return RHS_data["RHS"]
end

function field_corr(x, xhat)
    @assert size(x) == size(xhat)
    a = vec(x)      # flatten to 1D
    b = vec(xhat)
    return cor(a, b)  # Pearson r
end
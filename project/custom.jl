using WaterLily
import WaterLily: ∂, @loop, @inside, inside_u
using JLD2
using Random
using Statistics

function scalar_grad(field::AbstractArray)
    T = eltype(field); sz = size(field); N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function grad_p(flow::Flow)
    return scalar_grad(flow.p) .* flow.μ₀
end

function RHS(flow::Flow{N};λ=WaterLily.quick,kwargs...) where N
    RHS = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad_p(flow)
    return RHS
end

function remove_ghosts(snapshot::AbstractArray)
    return snapshot[2:end-1, 2:end-1, :, :]
end


function preprocess_data!(data; tmin=-1, tmax=-1, n_samples=-1, clip_bc=false, verbose=true)
    time = data["time"]

    # Resolve defaults
    tlo = (tmin == -1) ? first(time) : tmin
    thi = (tmax == -1) ? last(time)  : tmax
    ns  = (n_samples == -1) ? length(time) : n_samples

    # Select indices within [tlo, thi]
    selected = findall(t -> t >= tlo && t <= thi, time)
    if isempty(selected)
        verbose && @info "No samples in [$(tlo), $(thi)]. Nothing changed."
        return data
    end

    # Downsample to ns entries (clamped to available range)
    ns = clamp(ns, 1, length(selected))
    idx_in_selected = round.(Int, collect(range(1, length(selected), length=ns)))
    final_idx = selected[idx_in_selected]

    # Helper: downsample a key if present
    @inline function maybe_downsample!(dict, key, idx)
        if haskey(dict, key)
            val = dict[key]
            d = ndims(val)
            if d == 1
                dict[key] = val[idx]
            else
                leading = ntuple(_ -> Colon(), d - 1)
                dict[key] = val[leading..., idx]
            end
        end

        return nothing
    end

    # Apply downsampling
    for k in ("time", "Δt", "RHS", "flow", "μ₀", "u")
        maybe_downsample!(data, k, final_idx)
    end

    # Optional boundary clipping
    if clip_bc
        @inline function clip_time_series(ts)
            H, W, C, T = size(ts)
            temp = similar(ts, eltype(ts), H-2, W-2, C, T)
            @inbounds for i in axes(ts, 4)
                temp[:,:,:,i] = remove_ghosts(ts[:,:,:,i])
            end
            return temp
        end

        # replace entries in the dict with the clipped arrays
        data["RHS"] = clip_time_series(data["RHS"])
        haskey(data, "μ₀") && (data["μ₀"] = clip_time_series(data["μ₀"]))
        haskey(data, "u")  && (data["u"]  = clip_time_series(data["u"]))
    end

    if verbose
        @info "Downsampled to $(length(data["time"])) time steps."
        sz = (haskey(data, "RHS") && !isempty(data["RHS"])) ? size(data["RHS"]) : "—"
        @info "Input data size: $(sz)"
    end

    return data
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




function field_corr(x, xhat)
    @assert size(x) == size(xhat)
    a = vec(x)      # flatten to 1D
    b = vec(xhat)
    return cor(a, b)  # Pearson r
end
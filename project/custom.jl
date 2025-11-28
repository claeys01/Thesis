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
        haskey(data, "RHS") && (data["RHS"] = clip_time_series(data["RHS"]))
        haskey(data, "μ₀") && (data["μ₀"] = clip_time_series(data["μ₀"]))
        haskey(data, "u")  && (data["u"]  = clip_time_series(data["u"]))
    end

    if verbose
        @info "Downsampled to $(length(data["time"])) time steps."
        sz = (haskey(data, "u") && !isempty(data["u"])) ? size(data["u"]) : "—"
        @info "Input data size: $(sz)"
    end

    return data
end

ispow2(n::Integer) = n > 0 && (n & (n - 1)) == 0

function strain_field(u)
    all_dims = size(u)                    
    if !ispow2(all_dims[1]) 
        u = dropdims(remove_ghosts(u), dims = 4)
    end
    Tp = eltype(u)
    D = ndims(u) - 1
    dims= size(u)
    spatial = Tuple(size(u)[1:end-1])
    Sfield = zeros(Tp, dims..., D) 
    @loop Sfield[I,:,:] .= S(I, u) over I ∈ CartesianIndices(spatial)

    return Sfield   
end

function kinetic_energy_diffusion(u::AbstractArray; ν::Real=1.0)
    S = strain_field(u)
    ε = 2f0 * ν .* sum(S .^ 2, dims = (3, 4))  # sum over i,j
    ε = dropdims(ε, dims = (3, 4))
    return ε
end
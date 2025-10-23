# Normalizer struct + helpers
struct Normalizer
    μ      # per-channel mean (C,) on same device as data
    σ      # per-channel std (C,) on same device as data
    method::Symbol
end

"""
    normalize_batch(x; normalizer=nothing, method=:zscore, eps=1e-8f0)

Normalize a single sample (H,W,C) or a batch (H,W,C,N). Per-channel mean/std are computed
over spatial dims and samples (dims=(1,2,4)). If `normalizer` is `nothing` the per-channel
statistics are returned with the normalized data. All outputs (xn and normalizer μ/σ)
are Float32. `normalizer`, if provided, must match input channel count.
"""
function normalize_batch(x; normalizer=nothing, method::Symbol=:zscore, eps::Float32=Float32(1e-8))
    nd = ndims(x)
    if nd == 3
        x4 = reshape(x, size(x)..., 1)  # (H,W,C,1)
    elseif nd == 4
        x4 = x
    else
        throw(ArgumentError("normalize_batch expects 3-D or 4-D array, got ndims=$(nd)"))
    end

    C = size(x4, 3)

    if normalizer === nothing
        if method != :zscore
            throw(ArgumentError("only :zscore method implemented when computing normalizer"))
        end
        # per-channel mean/std across H, W and samples (dims 1,2,4)
        μ = Float32.(vec(mean(x4, dims=(1,2,4))))         # (C,)
        σ = Float32.(vec(std(x4, dims=(1,2,4)))) .+ eps   # (C,)
        normalizer = Normalizer(μ, σ, method)
    else
        μ = Float32.(normalizer.μ)
        σ = Float32.(normalizer.σ)
        if length(μ) != C || length(σ) != C
            throw(ArgumentError("normalizer channel count $(length(μ)) does not match data channels $C"))
        end
    end

    μr = reshape(μ, 1, 1, C, 1)   # (1,1,C,1)
    σr = reshape(σ, 1, 1, C, 1)

    if normalizer.method == :zscore
        xn = (x4 .- μr) ./ σr
        xn = Float32.(xn)
    else
        throw(ArgumentError("unsupported normalization method: $(normalizer.method)"))
    end

    return nd == 3 ? reshape(xn, size(x)...) : xn, normalizer
end

"""
    denormalize_batch(xn, normalizer)

Undo normalization produced by `normalize_batch`. Accepts 3-D or 4-D normalized data.
Denormalized output is Float32 and normalizer is expected to contain per-channel μ/σ.
"""
function denormalize_batch(xn, normalizer::Normalizer)
    nd = ndims(xn)
    if nd == 3
        xn4 = reshape(xn, size(xn)..., 1)
    elseif nd == 4
        xn4 = xn
    else
        throw(ArgumentError("denormalize_batch expects 3-D or 4-D normalized data, got ndims=$(nd)"))
    end

    C = size(xn4, 3)
    μ = Float32.(normalizer.μ)
    σ = Float32.(normalizer.σ)
    if length(μ) != C || length(σ) != C
        throw(ArgumentError("normalizer channel count $(length(μ)) does not match data channels $C"))
    end

    μr = reshape(μ, 1, 1, C, 1)
    σr = reshape(σ, 1, 1, C, 1)

    if normalizer.method == :zscore
        x = xn4 .* σr .+ μr
        x = Float32.(x)
    else
        throw(ArgumentError("unsupported normalization method: $(normalizer.method)"))
    end

    return nd == 3 ? reshape(x, size(xn)...) : x
end

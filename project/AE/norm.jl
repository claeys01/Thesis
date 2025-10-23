#!/usr/bin/env julia

using Statistics, Random

includet("../custom.jl")
"""
    normalize_snapshots(X; eps=1f-6)

Normalize a 4D tensor of fluid snapshots `X` with shape (H, W, C, N),
where H, W are spatial dimensions, C are channels (e.g. velocity components),
and N is the number of samples.

Performs per-channel z-score normalization:
    X_norm = (X - μ) / (σ + eps)

Returns:
    X_norm :: Array{Float32,4} — normalized data
    μ :: Array{Float32,4}      — per-channel mean (1,1,C,1)
    σ :: Array{Float32,4}      — per-channel std  (1,1,C,1)
    eps :: T                   — random small value to avoid division by zero

Use `denormalize_snapshots(X_norm, μ, σ)` to invert the operation.
"""
function normalize_snapshots(X::AbstractArray{T,4}; eps::T=1f-6) where {T<:Real}
    @assert ndims(X) == 4 "Input must be a 4D array (H, W, C, N)"
    X = Float32.(X)

    μ = mean(X; dims=(1,2,4))
    println(μ)
    σ = std(X; dims=(1,2,4))

    μ = reshape(μ, 1, 1, size(X,3), 1)
    σ = reshape(σ, 1, 1, size(X,3), 1)

    X_norm = (X .- μ) ./ (σ .+ eps)
    return X_norm, μ, σ
end

"""
    denormalize_snapshots(X_norm, μ, σ; eps=1f-6)

Inverse of `normalize_snapshots`.
"""
function denormalize_snapshots(X_norm::AbstractArray{T,4},
                               μ::AbstractArray{T,4},
                               σ::AbstractArray{T,4};
                               eps::T=1f-6) where {T<:Real}
    return X_norm .* (σ .+ eps) .+ μ
end

# ---------------------------------------------------------------------
# Demo section: create fake CFD snapshots and test normalization
# ---------------------------------------------------------------------

function main()
    Random.seed!(42)
    # H, W, C, N = 128, 128, 2, 500  # typical CFD shape
    X, _ = get_random_snapshots("/home/matth/Thesis/data/RHS_biot_data_arr.jld2")
    H, W, C, N = size(X)
        
    println("Generating random test data of size ($H,$W,$C,$N)...")

    # Normalize
    X_norm, μ, σ = normalize_snapshots(X)

    # Check that each channel is mean≈0, std≈1
    println("\n--- Normalization check ---")
    mean_channels = vec(mean(X_norm; dims=(1,2,4)))
    std_channels  = vec(std(X_norm; dims=(1,2,4)))
    println("Mean per channel ≈ 0: ", mean_channels)
    println("Std  per channel ≈ 1: ", std_channels)

    # Denormalize and verify reconstruction
    X_recon = denormalize_snapshots(X_norm, μ, σ)
    diff = maximum(abs.(X .- X_recon))
    println("\n--- Reconstruction check ---")
    println("Max |X - X_recon| = ", diff)
    println(diff < 1e-6 ? "✅ Passed!" : "⚠️ Too large reconstruction error!")
end

# Run automatically if file is executed directly
if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    main()
end

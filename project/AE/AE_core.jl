using Flux
using NNlib
using WaterLily
using Statistics
using MLUtils: DataLoader

includet("../utils/AE_normalizer.jl")

Base.@kwdef mutable struct Args
    η = 1e-3                    # learning rate
    λ = 1e-4                    # regularization paramater
    λdiv = 0                    # divergence loss weight
    λmask = 0
    batch_size = 32             # batch size
    downsample = -1             # amount of RHS used for training 
    epochs = 50               # number of epochs
    seed = 42                   # random seed
    n_reconstruct = 2           # sampling size for output    
    use_gpu = false             # use GPU
    clip_bc = true              # removes the ghost cells from the snapshot
    input_dim = (128, 128, 2)   # flow field size
    split = 0.2
    stride = 2
    padding = 1
    latent_dim = 8^3           # latent dimension
    C_conv = 8                  # first amount of channels for convs
    verbose_freq = 5            # logging for every verbose_freq iterations
    normalize = true            # normalise training data
    save_path = "data/models"   # results path
    data_path = "data/datasets/128_RHS_biot_data_arr_force_period.jld2"
end


function get_data(batch_size, path; tmin=-1, tmax=-1, n_samples=500, normalize=false, clip_bc=true, split=0.2)
    @load path RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=n_samples, clip_bc=clip_bc)

    # construct data array (H,W,C,N) and cast to Float32
    X = cat(RHS_data["RHS"]...; dims=4)
    X = Float32.(X)

    # compute normalizer on full dataset (returned always)
    X_norm, normalizer = normalize_batch(X; normalizer=nothing)

    # split indices into train / val
    N = size(X, 4)
    if N < 2
        error("get_data: need at least 2 samples to create train/validation split (got $N)")
    end
    nval = max(1, Int(round(split * N)))
    nval = min(nval, N-1)   # ensure at least one train sample
    @info "Data_split into $(N-nval) training  and $(nval) validation snapshots"
    perm = randperm(N)
    val_idx = perm[1:nval]
    train_idx = perm[(nval+1):end]

    # get the μ₀ field of the training data
    μ₀ = cat(RHS_data["μ₀"]...; dims=4)
    μ₀ = Float32.(μ₀)

    # build DataLoaders
    if normalize
        train_data = (X_norm[:, :, :, train_idx], μ₀[:, :, :, train_idx])
        val_data = (X_norm[:, :, :, val_idx], μ₀[:, :, :, val_idx])
        train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
        val_loader   = DataLoader(val_data,   batchsize=batch_size, shuffle=true)
    else
        train_data = (X[:, :, :, train_idx], μ₀[:, :, :, train_idx])
        val_data = (X[:, :, :, val_idx], μ₀[:, :, :, val_idx])
        train_loader = DataLoader(train_data, batchsize=batch_size, shuffle=true)
        val_loader   = DataLoader(val_data,   batchsize=batch_size, shuffle=true)
    end

    return train_loader, val_loader, normalizer
end


struct Encoder 
    layers::Chain
end


Flux.@layer Encoder

Encoder(input_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
    H, W, C = input_size

    convpart = Chain(
        Conv((3,3), C           => C_next,   identity; pad=padding, stride=stride), relu,
        MaxPool((2,2)),
        Conv((3,3), C_next      => 2C_next,  identity; pad=padding, stride=stride), relu,
        MaxPool((2,2)),
        Conv((3,3), 2C_next     => 4C_next,  identity; pad=padding, stride=stride), relu,
        MaxPool((2,2)),
        Conv((3,3), 4C_next     => 8C_next,  identity; pad=padding, stride=stride), relu,
        MaxPool((2,2)),
        Flux.flatten
    )
    dummy = zeros(Float32, H, W, C, 1)
    flat = convpart(dummy)
    dense_in = size(flat, 1)
    if verbose
        @info "Initialize Encoder with $(dense_in) connected nodes and $(latent_dim) latent dimensions"
    end
    return Encoder(Chain(convpart, Dense(dense_in, latent_dim)))
end

# Encoder(input_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
#     H, W, C = input_size

#     convpart = Chain(
#         # level 1: H,W -> H/2,W/2
#         Conv((3,3), C        => C_next,   relu; pad=padding, stride=2),
#         Conv((3,3), C_next   => C_next,   relu; pad=padding, stride=1),

#         # level 2: H/2 -> H/4
#         Conv((3,3), C_next   => 2C_next,  relu; pad=padding, stride=2),
#         Conv((3,3), 2C_next  => 2C_next,  relu; pad=padding, stride=1),

#         # level 3: H/4 -> H/8
#         Conv((3,3), 2C_next  => 4C_next,  relu; pad=padding, stride=2),
#         Conv((3,3), 4C_next  => 4C_next,  relu; pad=padding, stride=1),

#         # level 4: H/8 -> H/16
#         Conv((3,3), 4C_next  => 8C_next,  relu; pad=padding, stride=2),
#         Conv((3,3), 8C_next  => 8C_next,  relu; pad=padding, stride=1),

#         Flux.flatten
#     )
#     dummy = zeros(Float32, H, W, C, 1)
#     flat = convpart(dummy)
#     dense_in = size(flat, 1)
#     if verbose
#         @info "Initialize Encoder with $(dense_in) connected nodes and $(latent_dim) latent dimensions"
#     end
#     return Encoder(Chain(convpart, Dense(dense_in, latent_dim)))
# end


function (encoder::Encoder)(x)
    z = encoder.layers(x)
    return z
end


struct Decoder 
    layers::Chain
end

function (decoder::Decoder)(x)
    x̂ = decoder.layers(x)
    return x̂
end

Flux.@layer Decoder


# Decoder(output_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4, verbose::Bool=true) = begin
#     H, W, C = output_size
#     # after four 2x2 downsamples: h_out = H ÷ 16, w_out = W ÷ 16

#     h_lat = div(H, 16)
#     w_lat = div(W, 16)
#     channels_mid = 8 * C_next
#     dense_len = h_lat * w_lat * channels_mid
    
#     verbose && @info "Initialize Decoder with $(dense_len) connected nodes and $(latent_dim) latent dimensions"
    

#     return Decoder(Chain(
#         Dense(latent_dim, dense_len),
#         x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2)),
#         ConvTranspose((2, 2), 8*C_next => 4*C_next, relu; stride=(2, 2), pad=(0, 0)),
#         ConvTranspose((2, 2), 4*C_next => 2*C_next, relu; stride=(2, 2), pad=(0, 0)),
#         ConvTranspose((2, 2), 2*C_next => C_next,   relu; stride=(2, 2), pad=(0, 0)),
#         ConvTranspose((2, 2), C_next   => C, relu; stride=(2, 2), pad=(0, 0)),
#         ConvTranspose((3, 3), C => C, identity; stride=(1, 1), pad=(1, 1))
#     ))
# end

function upsample2(x)
    H, W, C, N = size(x)
    return NNlib.upsample_bilinear(x; size=(2H, 2W))
end
Decoder(output_size::Tuple{Int,Int,Int}, latent_dim::Int; C_next::Int=4, verbose::Bool=true) = begin
    H, W, C = output_size
    h_lat, w_lat = div(H,16), div(W,16)
    channels_mid = 8*C_next
    dense_len = h_lat * w_lat * channels_mid
    verbose && @info "Initialize Decoder (upsample+conv) with $(dense_len) nodes, $(latent_dim) latent dims"

    return Decoder(Chain(
        Dense(latent_dim, dense_len),
        x -> reshape(x, h_lat, w_lat, channels_mid, size(x,2)),

        # stage 1: H/16 → H/8
        x -> upsample2(x),
        Conv((3,3), 8C_next => 4C_next; pad=1), gelu,

        # stage 2: H/8 → H/4
        x -> upsample2(x),
        Conv((3,3), 4C_next => 2C_next; pad=1), gelu,

        # stage 3: H/4 → H/2
        x -> upsample2(x),
        Conv((3,3), 2C_next => 1C_next; pad=1), gelu,

        # stage 4: H/2 → H
        x -> upsample2(x),
        Conv((3,3), C_next => C; pad=1),     # linear output (no relu!)
        Conv((3,3), C => C; pad=1)           # small anti-alias / smoothing
    ))
end

function reconstruct(enc::Encoder, dec::Decoder, x)
    z = enc(x)
    return dec(z)
end

function check_ae_dims(encoder, decoder, x; device=Flux.get_device("CPU"))
    x_dev = x |> device
    ŷ = reconstruct(encoder, decoder, x_dev)
    return size(x_dev) == size(ŷ), size(x_dev), size(ŷ)
end


"""
    divergence_ad(field; dx=1.0, dy=1.0)

Zygote-safe divergence of a 2D vector field stored as (Nx, Ny, 2).
Assumes periodic boundaries (via circshift).

- field[:,:,1] = u(x,y)
- field[:,:,2] = v(x,y)
Returns (Nx, Ny).
"""
function divergence(field; dx=1.0, dy=1.0)
    u = field[:, :, 1]
    v = field[:, :, 2]
    du_dx = (circshift(u, (-1, 0)) .- circshift(u, (1, 0))) ./ (2dx)
    dv_dy = (circshift(v, (0, -1)) .- circshift(v, (0, 1))) ./ (2dy)
    du_dx .+ dv_dy
end

function divergence_field(u; mean=false, max=false)
    if ndims(u) == 4
        H, W, C, N = size(u)
        # Build a list of H×W matrices (one per sample) without mutating.
        mats = [divergence_field(view(u, :, :, :, n); mean=false, max=false) for n in 1:N]
        σ = cat(mats...; dims=3)  # result has shape (H, W, N)
        if mean
            return mean(σ)
        elseif max
            return maximum(σ)
        else
            return σ
        end
    elseif ndims(u) == 3
        H, W, C = size(u)
        σ = divergence(u)
        if mean
            return mean(σ)
        elseif max
            return maximum(σ)
        else
            return σ
        end
    else
        throw(ArgumentError("divergence_field expects a 3-D (H,W,C) or 4-D (H,W,C,N) array; got ndims=$(ndims(u))"))
    end
end


# losses
recon_loss(ŷ, x) = mean(abs2, ŷ .- x)                # MSE
div_loss_L2(u) = mean(abs2, divergence_field(u))     # L2 of divergence field

function masked_loss(x, x̂, μ₀)
    outside = μ₀
    inside = 1f0 .- μ₀
    # Lrec = recon_loss(x̂ .* outside, x)
    Lrec = mean(abs2, (x̂ .- x) .* outside)  
    Linside = mean(abs2, x̂ .* inside)
    # Linside = 0
    return Lrec, Linside
end


# combined total loss (x̂ = decoder(z) or ae(x))
function total_loss(encoder::Encoder, decoder::Decoder, x::AbstractArray, μ₀::AbstractArray; λdiv=0, λmask=0)
    x̂ = reconstruct(encoder, decoder, x)
    # Lrec = recon_loss(x̂, x)

    if λmask != 0
        Lrec, Linside = masked_loss(x, x̂, μ₀)
    else
        Lrec = recon_loss(x̂, x)
        Linside = zero(eltype(Lrec))
    end
    L2div = zero(eltype(Lrec))
    if λdiv != 0
        try
            # If divergence computation fails on GPU we skip it (avoid CPU/GPU mix)
            L2div = div_loss_L2(x̂)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
            L2div = zero(eltype(Lrec))
        end
    end
    

    return Lrec + λdiv * L2div + λmask * Linside, (Lrec, Linside, L2div)
end


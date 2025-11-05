using Flux
using NNlib
using WaterLily
using Statistics
using Zygote
using MLUtils: DataLoader

includet("../utils/AE_normalizer.jl")

Base.@kwdef mutable struct Args
    η = 5e-3                    # learning rate
    λ = 1e-4                    # regularization paramater
    λdiv = 0                    # divergence loss weight
    λmask = 0                   # weight of body mask loss
    loss = :L1               # loss function for reconstruction loss (:L1, :L2, :charb)
    batch_size = 32             # batch size
    downsample = 100            # amount of RHS used for training 
    epochs = 100                # number of epochs
    seed = 42                   # random seed
    n_reconstruct = 2           # sampling size for output    
    use_gpu = false             # use GPU
    clip_bc = true              # removes the ghost cells from the snapshot
    input_dim = (128, 128, 4)   # flow field size with μ₀ concatenated
    output_dim = (128, 128, 2)  # size of reconstructed RHS field
    split = 0.2
    stride = 1
    padding = 1
    latent_dim = 8^2            # latent dimension
    C_conv = 8                  # first amount of channels for convs
    verbose_freq = 5            # logging for every verbose_freq iterations
    normalize = true            # normalise training data
    save_path = "data/models"   # results path
    data_path = "data/datasets/128_RHS_biot_data_arr_force_period.jld2"
end


function get_data(batch_size, path; tmin=-1, tmax=-1, n_samples=500,
    normalize=false, clip_bc=true, split=0.2)

    @load path RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax,
        n_samples=n_samples, clip_bc=clip_bc)

    # X :: (H,W,2,N)
    X = cat(RHS_data["RHS"]...; dims=4)
    X = Float32.(X)

    # μ₀ :: (H,W,1,N), 1 outside; 0 inside
    μ₀ = cat(RHS_data["μ₀"]...; dims=4)
    μ₀ = Float32.(μ₀)

    # normalizer from X only (physics channels)
    X_norm, normalizer = normalize_batch(X; normalizer=nothing)

    # indices
    N = size(X, 4)
    if N < 2
        error("get_data: need at least 2 samples to create train/validation split (got $N)")
    end
    nval = max(1, Int(round(split * N)))
    nval = min(nval, N - 1)
    @info "Data_split into $(N-nval) training and $(nval) validation snapshots"
    perm = randperm(N)
    val_idx = perm[1:nval]
    train_idx = perm[(nval+1):end]

    if normalize
        Xin_train = cat(X_norm[:, :, :, train_idx], μ₀[:, :, :, train_idx]; dims=3) # (u,v,mask)
        Xin_val = cat(X_norm[:, :, :, val_idx], μ₀[:, :, :, val_idx]; dims=3)

        Xtarget_train = X_norm[:, :, :, train_idx]  # (u,v)
        Xtarget_val = X_norm[:, :, :, val_idx]

        μ₀_train = μ₀[:, :, :, train_idx]
        μ₀_val = μ₀[:, :, :, val_idx]
    else
        Xin_train = cat(X[:, :, :, train_idx], μ₀[:, :, :, train_idx]; dims=3)
        Xin_val = cat(X[:, :, :, val_idx], μ₀[:, :, :, val_idx]; dims=3)

        Xtarget_train = X[:, :, :, train_idx]
        Xtarget_val = X[:, :, :, val_idx]

        μ₀_train = μ₀[:, :, :, train_idx]
        μ₀_val = μ₀[:, :, :, val_idx]
    end

    train_loader = DataLoader((Xin_train, Xtarget_train, μ₀_train),
        batchsize=batch_size, shuffle=true)
    val_loader = DataLoader((Xin_val, Xtarget_val, μ₀_val),
        batchsize=batch_size, shuffle=true)

    return train_loader, val_loader, normalizer
end


struct Encoder
    layers::Chain
end


Flux.@layer Encoder

Encoder(input_size::Tuple{Int,Int,Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
    H, W, C = input_size

    convpart = Chain(
        Conv((3, 3), C => C_next, identity; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), C_next => 2C_next, identity; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 2C_next => 4C_next, identity; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 4C_next => 8C_next, identity; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
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



struct Decoder
    layers::Chain
end


Flux.@layer Decoder

function upsample2(x)
    H, W, _, _ = size(x)
    return NNlib.upsample_bilinear(x; size=(2H, 2W))
end
Decoder(output_size::Tuple{Int,Int,Int}, latent_dim::Int; C_next::Int=4, verbose::Bool=true) = begin
    H, W, C = output_size
    h_lat, w_lat = div(H, 16), div(W, 16)
    channels_mid = 8 * C_next
    dense_len = h_lat * w_lat * channels_mid
    verbose && @info "Initialize Decoder (upsample+conv) with $(dense_len) nodes, $(latent_dim) latent dims"

    return Decoder(Chain(
        Dense(latent_dim, dense_len),
        x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2)),

        # stage 1: H/16 → H/8
        x -> upsample2(x),
        Conv((3, 3), 8C_next => 4C_next; pad=1), gelu,

        # stage 2: H/8 → H/4
        x -> upsample2(x),
        Conv((3, 3), 4C_next => 2C_next; pad=1), gelu,

        # stage 3: H/4 → H/2
        x -> upsample2(x),
        Conv((3, 3), 2C_next => 1C_next; pad=1), gelu,

        # stage 4: H/2 → H
        x -> upsample2(x),
        Conv((3, 3), C_next => C; pad=1),     # linear output (no relu!)
        Conv((3, 3), C => C; pad=1)           # small anti-alias / smoothing
    ))
end


function (encoder::Encoder)(x)
    z = encoder.layers(x)
    return z
end

function (decoder::Decoder)(z, μ₀)
    x̂ = decoder.layers(z)
    return x̂ .* μ₀
end

function reconstruct(enc::Encoder, dec::Decoder, x, μ₀)
    z = enc(x)
    return dec(z, μ₀)
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

function recon_loss(x, x̂; loss=:L2)
    Lrec = 0.0
    if loss == :L1
        Lrec = mean(abs, x̂ .- x)
    elseif loss == :L2
        Lrec = mean(abs2, x̂ .- x)
    elseif loss == :charb
        Lrec = Lrec_charbonnier(x, x̂)
    end
    return Lrec
end

function Lrec_charbonnier(x, x̂; eps=1f-10)
    Δ = (x̂ .- x)
    mean(sqrt.(Δ .^ 2 .+ eps^2))
end


function Lrec_charbonnier_mask(x, x̂, μ₀; eps=1f-4)
    Δ = (x̂ .- x) .* μ₀
    mean(sqrt.(Δ .^ 2 .+ eps^2))
end

function masked_loss(x, x̂, μ₀; loss=:L2)
    outside = μ₀
    inside = 1f0 .- μ₀
    boundary = outside .* inside
    Lrec = Lrec_charbonnier_mask(x, x̂, outside; eps=1f-4)
    Linside = mean(abs, (x̂ .- x) .* boundary)
    return Lrec, Linside
end



"""
    field_corr(x, xhat)

Return Pearson correlation coefficient between x and xhat.
Both can be any N-D array of equal size.
"""
function field_corr(x, x̂)
    @assert size(x) == size(xhat)
    a = vec(x)      # flatten to 1D
    b = vec(x̂)
    return cor(a, b)  # Pearson r
end


function batch_corrs(x, x̂)
    @assert size(x) == size(x̂)
    nx, ny, nchan, nbatch = size(x)
    rbatch = zeros(Float32, nchan, nbatch)
    for b in 1:nbatch
        for c in 1:nchan
            rbatch[c, b] = cor(vec(view(x, :, :, c, b)),
                vec(view(x̂, :, :, c, b)))
        end
    end
    return Float32.(mean(rbatch; dims=2)[:])  # average over batch dimension, as Float32
end

Zygote.@nograd batch_corrs

# combined total loss (x̂ = decoder(z) or ae(x))
function total_loss(encoder::Encoder,
    decoder::Decoder,
    x_in::AbstractArray,        # (u,v,mask)
    x_target::AbstractArray,    # (u,v)
    μ₀::AbstractArray;          # (H,W,1,B)  1 outside / 0 inside
    loss=:L2,
    λdiv=0f0,
    λmask=0f0)
    x̂ = reconstruct(encoder, decoder, x_in,  μ₀)

    corrs = batch_corrs(x_target, x̂)

    Linside = zero(Float32)
    L2div = zero(Float32)

    if λmask != 0
        Lrec, Linside = masked_loss(x_target, x̂, μ₀; loss=loss)
    elseif λdiv != 0
        try
            # If divergence computation fails on GPU we skip it (avoid CPU/GPU mix)
            L2div = div_loss_L2(x̂)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
        end
    else
        Lrec = recon_loss(x_target, x̂; loss=loss)
    end

    return Lrec + λdiv * L2div + λmask * Linside, (Lrec, Linside, L2div), corrs
end


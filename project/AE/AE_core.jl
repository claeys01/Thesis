using Flux
using Flux: glorot_uniform, Conv, ConvTranspose, Dense, Chain, relu, MaxPool
using WaterLily
using Random
using Statistics
using ProgressMeter: Progress, next!
using MLUtils: DataLoader
using Zygote

includet("../utils/AE_normalizer.jl")


function get_data(batch_size, path; tmin=-1, tmax=-1, n_samples=500, normalize=false)
    @load path RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=n_samples, clip_bc=true)
    X = cat(RHS_data["RHS"]...; dims=4)   # now X has shape (H,W,C,N)
    X = Float32.(X)
    X_normalized, normalizer = normalize_batch(X; normalizer=nothing)
    if normalize
        loader = DataLoader(X_normalized, batchsize=batch_size, shuffle=true)
    else
        loader = DataLoader(X, batchsize=batch_size, shuffle=true)
    end
    return loader, normalizer
end


struct Encoder 
    layers::Chain
end


Flux.@layer Encoder

Encoder(input_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2) = begin
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
    @info "Initialize Encoder with $(dense_in) connected nodes and $(latent_dim) latent dimensions"
    return Encoder(Chain(convpart, Dense(dense_in, latent_dim)))
end

# Encoder(input_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2) = begin
#     H, W, C = input_size

#     convpart = Chain(
#         Conv((3,3), C           => C_next,   identity; pad=padding, stride=stride), relu,
#         Conv((3,3), C_next      => 2C_next,  identity; pad=padding, stride=stride), relu,
#         Conv((3,3), 2C_next     => 4C_next,  identity; pad=padding, stride=stride), relu,
#         Conv((3,3), 4C_next     => 8C_next,  identity; pad=padding, stride=stride), relu,
#         Flux.flatten
#     )
#     dummy = zeros(Float32, H, W, C, 1)
#     flat = convpart(dummy)
#     dense_in = size(flat, 1)
#     @info "Initialize Encoder with $(dense_in) connected nodes and $(latent_dim) latent dimensions"
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


Decoder(output_size::Tuple{Int,Int, Int}, latent_dim::Int; C_next::Int=4) = begin
    H, W, C = output_size
    # after four 2x2 downsamples: h_out = H ÷ 16, w_out = W ÷ 16
    # if H % 16 != 0 || W % 16 != 0
        # throw(ArgumentError("Input of encoder needs to be deviseable by 16, please exclude ghost cells"))
    # end

    h_lat = div(H, 16)
    w_lat = div(W, 16)
    channels_mid = 8 * C_next
    dense_len = h_lat * w_lat * channels_mid
    @info "Initialize Dencoder with $(dense_len) connected nodes and $(latent_dim) latent dimensions"

    return Decoder(Chain(
        Dense(latent_dim, dense_len),
        x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2)),
        ConvTranspose((2, 2), 8*C_next => 4*C_next, relu; stride=(2, 2), pad=(0, 0)),
        ConvTranspose((2, 2), 4*C_next => 2*C_next, relu; stride=(2, 2), pad=(0, 0)),
        ConvTranspose((2, 2), 2*C_next => C_next,   relu; stride=(2, 2), pad=(0, 0)),
        ConvTranspose((2, 2), C_next   => C, relu; stride=(2, 2), pad=(0, 0)),
        ConvTranspose((3, 3), C => C, identity; stride=(1, 1), pad=(1, 1))
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
div_diff_loss(ŷ, x) = mean(abs2, divergence_field(ŷ) .- divergence_field(x))


# combined total loss (ŷ = decoder(z) or ae(x))
function total_loss(encoder, decoder, x; λdiv=0, λdiff=0)
    ŷ = reconstruct(encoder, decoder, x)
    Lrec = recon_loss(ŷ, x)
    L2div = zero(eltype(Lrec))
    L2div_diff = zero(eltype(Lrec))
    if λdiv != 0
        try
            # If divergence computation fails on GPU we skip it (avoid CPU/GPU mix)
            L2div = div_loss_L2(ŷ)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
            L2div = zero(eltype(Lrec))
        end
    end
    if λdiff != 0
        try 
            div_diff = divergence_field(ŷ) .- divergence_field(x)
            L2div_diff = mean(abs2, div_diff)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
            L2div_diff = zero(eltype(Lrec))
        end
    end
    return Lrec + λdiv * L2div + λdiff * L2div_diff, (Lrec, L2div, L2div_diff)
end

Base.@kwdef mutable struct Args
    η = 1e-3                    # learning rate
    λ = 1e-4                    # regularization paramater
    λdiv = 0                    # divergence loss weight
    λdiff = 0                   # divergence difference weight
    batch_size = 256             # batch size
    downsample = 1500           # amount of RHS used for training 
    epochs = 100                  # number of epochs
    seed = 42                   # random seed
    n_reconstruct = 2           # sampling size for output    
    use_gpu = false             # use GPU
    input_dim = (128, 128, 2)   # flow field size
    stride = 1
    padding = 1
    latent_dim = 8^3             # latent dimension
    C_conv = 8                  # first amount of channels for convs
    verbose_freq = 5            # logging for every verbose_freq iterations
    normalize = true           # normalise training data
    save_path = "data/models"   # results path
    data_path = "data/RHS_biot_data_arr.jld2"
end
using Lux
using NNlib
using Statistics

Base.@kwdef mutable struct LuxArgs
    η = 5e-3                    # learning rate
    λ = 1e-4                    # regularization paramater
    λdiv = 0                    # divergence loss weight
    λmask = 0                   # weight of body mask loss
    loss = :L1               # loss function for reconstruction loss (:L1, :L2, :charb)
    batch_size = 32             # batch size
    downsample = -1            # amount of RHS used for training 
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


struct Encoder
    layers::Chain
end


Encoder(input_size::Tuple{Int,Int,Int}, latent_dim::Int; C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
    H, W, C = input_size

    convpart = Chain(
        # NOTE: no activation argument in Lux.Conv; add relu separately
        Conv((3, 3), C => C_next; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), C_next => 2C_next; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 2C_next => 4C_next; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 4C_next => 8C_next; pad=padding, stride=stride), relu,
        MaxPool((2, 2)),
        FlattenLayer()  # instantiate
    )
    dummy = zeros(Float32, H, W, C, 1)

    # initialize convpart params/state and run forward to infer flattened size
    rng = Xoshiro(0)
    ps_conv, st_conv = Lux.setup(rng, convpart)
    flat, st_conv = convpart(dummy, ps_conv, st_conv)
    dense_in = size(flat, 1)

    verbose && @info "Initialize Encoder with $(dense_in) connected nodes and $(latent_dim) latent dimensions"

    return Encoder(Chain(convpart, Dense(dense_in, latent_dim)))
end


struct Decoder
    layers::Chain
end


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

function (decoder::Decoder)(z)
    x̂ = decoder.layers(z)
    return x̂
end

function reconstruct(enc::Encoder, dec::Decoder, x)
    z = enc(x)
    return dec(z)
end


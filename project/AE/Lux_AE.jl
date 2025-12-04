using Lux
using NNlib
using Statistics
using MLUtils
using Zygote


includet("../utils/AE_normalizer.jl")
includet("../custom.jl")
includet("../utils/SimDataTypes.jl")
using .SimDataTypes

Base.@kwdef mutable struct LuxArgs
    η = 5e-3                    # learning rate
    λ = 1e-4                    # regularization paramater
    Autodiff = AutoZygote()
    λdiv = 0                    # divergence loss weight
    λmask = 0                   # weight of body mask loss
    loss = :L1                  # loss function for reconstruction loss (:L1, :L2, :charb)
    batch_size = 32             # batch size
    downsample = -1             # amount of data used for training 
    n_periods = 3               # amount of shedding periods to use for training data
    epochs = 50                # number of epochs
    seed = 42                   # random seed
    n_reconstruct = 2           # sampling size for output   
    test_loss = false
    test_downsample = 300
    field = "u"
    use_gpu = false             # use GPU
    clip_bc = true              # removes the ghost cells from the snapshot
    input_dim = (2^8, 2^8, 4)   # flow field size with μ₀ concatenated
    output_dim = (2^8, 2^8, 2)  # size of reconstructed RHS field
    split = 0.2
    stride = 1
    padding = 1
    latent_dim = 16            # latent dimension
    hidden_dim = 256
    C_conv = 8                  # first amount of channels for convs
    normalize = true            # normalise training data
    save_path = "data/Lux_models"   # results path
    data_path = "data/datasets/RE2500/2e8/U_128_period.jld2"
    full_data_path = "data/datasets/RE2500/2e8/U_128_full.jld2"
end

function get_data_in(X, μ₀; idx=nothing)
    if isnothing(idx)
        Xin = cat(X, μ₀; dims=3)
    else
        Xin, X, μ₀ = cat(X[:, :, :, idx], μ₀[:, :, :, idx]; dims=3), X[:, :, :, idx], μ₀[:, :, :, idx]
    end
    return (Xin, X, μ₀)
end


function load_simdata(path)
    @load path simdata
    return simdata::SimData
end

function downsample_equal(v::AbstractVector, M::Integer)
    N = length(v)
    M ≤ N || @warn "Cannot downsample to $M entries from $N points, returning $N points"
    M = clamp(M, 1, N)
    idx = round.(Int, range(1, N, length = M))
    return v[idx]
end


function get_data(batch_size, path;t_training=10, n_training=500, split=0.2, verbose=true)


    simdata = load_simdata(path)
    preprocess_data!(simdata; verbose=verbose)

    # X :: (H,W,C,N)
    X  = simdata.u
    μ₀ = simdata.μ₀

    # normalizer from X only (physics channels)
    _, normalizer = normalize_batch(X; normalizer=nothing)


    N = size(X, 4)

    if N < 2
        error("get_data: need at least 2 samples to create train/validation split (got $N)")
    end
    
    train_idxs_full = findall(t -> t < t_training, simdata.time)
    train_idx_combined = downsample_equal(train_idxs_full, n_training)
    
    # Split into train / val by downsampling evenly from the combined pool
    n_val = clamp(round(Int, length(train_idx_combined) * split), 0, length(train_idx_combined))

    if n_val == 0
        @warn "No validation data created"
        val_idx = Int[]
        train_idx = train_idx_combined
    else
        # Downsample evenly to get validation indices
        val_idx = downsample_equal(train_idx_combined, n_val)
        # Remove validation indices from train
        train_idx = setdiff(train_idx_combined, val_idx)
    end

    test_idx = collect(last(train_idx_combined)+1 : N)

    plt = train_force_plot(simdata; train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    display(plt)

    # DataLoaders over indices only (lightweight)
    train_loader = DataLoader(train_idx; batchsize=batch_size, shuffle=true)
    val_loader   = DataLoader(val_idx;   batchsize=batch_size, shuffle=false)
    test_loader  = DataLoader(test_idx;  batchsize=batch_size, shuffle=false)

    # return X, μ₀, normalizer 
    return train_loader, val_loader, test_loader, X, μ₀, normalizer
end



struct Encoder{L} <: AbstractLuxWrapperLayer{:layers}
    layers::L
end


Encoder(input_size::Tuple{Int,Int,Int}, latent_dim::Int; hidden_dim=256, C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
    H, W, C = input_size

    convpart = Chain(
        # NOTE: no activation argument in Lux.Conv; add relu separately
        Conv((3, 3), C => C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
        MaxPool((2, 2)),
        Conv((3, 3), C_next => 2C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 2C_next => 4C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
        MaxPool((2, 2)),
        Conv((3, 3), 4C_next => 8C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
        MaxPool((2, 2)),
        FlattenLayer()  # instantiate
    )
    dummy = zeros(Float32, H, W, C, 1)

    # initialize convpart params/state and run forward to infer flattened size
    rng = Xoshiro(0)
    ps_conv, st_conv = Lux.setup(rng, convpart)
    flat, st_conv = convpart(dummy, ps_conv, st_conv)
    dense_in = size(flat, 1)

    verbose && @info "Initialize Encoder with $(dense_in) → $(hidden_dim) → $(latent_dim) bottleneck"

    layers = Chain(
        convpart,
        Dense(dense_in, hidden_dim), relu,
        Dense(hidden_dim, latent_dim)      # latent_dim = 16 later
    )

    return Encoder(layers)
end

function (encoder::Encoder)(x, ps, st)
    z, st_new = encoder.layers(x, ps, st)
    return z, st_new
end

struct Decoder{L} <: AbstractLuxWrapperLayer{:layers}
    layers::L
end


function upsample2(x)
    H, W, _, _ = size(x)
    return NNlib.upsample_bilinear(x; size=(2H, 2W))
end


Decoder(output_size::Tuple{Int,Int,Int}, latent_dim::Int; hidden_dim=256, C_next::Int=4, verbose::Bool=true) = begin
    H, W, C = output_size
    h_lat, w_lat = div(H, 16), div(W, 16)
    channels_mid = 8 * C_next
    dense_len = h_lat * w_lat * channels_mid
    verbose && @info "Initialize Decoder (upsample+conv) with $(latent_dim) → $(hidden_dim) → $(dense_len)"

    layers = Chain(
        Dense(latent_dim, hidden_dim), gelu,
        Dense(hidden_dim, dense_len),
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
    )

    return Decoder(layers)
end

function (dec::Decoder)(z, ps, st)
    x̂, st_new = dec.layers(z, ps, st)
    return x̂, st_new
end

struct AE{E, D} <: AbstractLuxContainerLayer{(:encoder, :decoder)}
    encoder::E
    decoder::D
end

AE(enc::Encoder, dec::Decoder) = AE{typeof(enc), typeof(dec)}(enc, dec)

function (m::AE)(x, ps, st)
    # encoder pass
    z, st_enc = m.encoder(x, ps.encoder, st.encoder)
    # decoder pass
    x̂, st_dec = m.decoder(z, ps.decoder, st.decoder)
    # return reconstruction + updated state tree
    return x̂, (encoder = st_enc, decoder = st_dec)
end

# losses
# recon_loss(x, x̂) = mean(abs2, x̂ .- x)                # MSE
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

function total_loss(m::AE, ps, st, 
    x_in::AbstractArray,        # (u,v,mask)
    x_target::AbstractArray,    # (u,v)
    μ₀::AbstractArray;          # (H,W,1,B)  1 outside / 0 inside
    loss=:L1,
    λdiv=0f0,
    λmask=0f0)

    # 1) Forward pass through AE (Lux-style)
    x̂, st2 = m(x_in, ps, st)

    corrs = batch_corrs(x_target, x̂)

    # 3) Mask out body region
    x̂ = x̂ .* μ₀

    # 4) Reconstruction + optional extra losses
    Linside = 0f0
    L2div   = 0f0
    Lrec    = 0f0

    if λmask != 0f0
        Lrec, Linside = masked_loss(x_target, x̂, μ₀; loss=loss)
    elseif λdiv != 0f0
        try
            # If divergence computation fails on GPU we skip it (avoid CPU/GPU mix)
            L2div = div_loss_L2(x̂)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
        end
        Lrec = recon_loss(x_target, x̂; loss=loss)
    else
        Lrec = recon_loss(x_target, x̂; loss=loss)
    end

    L = Lrec + λdiv * L2div + λmask * Linside

    return L, st2, (Lrec, Linside, L2div, corrs)
end


"""
    load_trained_AE(checkpoint_path; device=cpu_device(), return_params=false)

Load a saved Lux checkpoint (expects keys "ps","st","args", "normalizer" optional).
Returns (encoder, decoder, ae) by default. If `return_params=true` returns
(encoder, decoder, ae, ps, st) where `ps` and `st` have been moved to `device`.
"""
function load_trained_AE(checkpoint_path::String; device=cpu_device(), return_params::Bool=false)
    checkpoint = JLD2.load(checkpoint_path)

    ps = get(checkpoint, "ps", nothing)
    st = get(checkpoint, "st", nothing)
    args_dict = checkpoint["args"]
    args = LuxArgs(; args_dict...)

    # infer device from saved args if caller left default cpu_device()
    if device === cpu_device() && hasproperty(args, :use_gpu)
        device = args.use_gpu ? gpu_device() : cpu_device()
    end

    # reinstantiate model architecture
    enc = Encoder(args.input_dim, args.latent_dim; hidden_dim=args.hidden_dim, C_next=args.C_conv, padding=args.padding, stride=args.stride, verbose=false)
    dec = Decoder(args.output_dim, args.latent_dim; hidden_dim=args.hidden_dim, C_next=args.C_conv, verbose=false)
    ae = AE(enc, dec)

    # move parameter/state trees to device if present
    if ps !== nothing
        ps = device(ps)
    end
    if st !== nothing
        st = device(st)
    end

    return return_params ? (enc, dec, ae, ps, st) : (enc, dec, ae)
end

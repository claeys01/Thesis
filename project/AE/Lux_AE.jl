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
    η = 1e-4                    # learning rate
    λ = 3e-4                    # regularization paramater
    Autodiff = AutoZygote()
    λdiv = 0                    # divergence loss weight
    λmask = 0                   # weight of body mask loss
    loss = :L1                  # loss function for reconstruction loss (:L1, :L2, :charb)
    batch_size = 32             # batch size
    t_training = 16.603
    train_downsample = 200     # amount of data used for training 
    test_downsample = 100
    split = 0.2
    n_periods = 3               # amount of shedding periods to use for training data
    epochs = 2                  # number of epochs
    seed = 42                   # random seed
    n_reconstruct = 2           # sampling size for output   
    test_loss = true
    field = "u"
    use_gpu = false             # use GPU
    clip_bc = true              # removes the ghost cells from the snapshot
    input_dim = (2^8, 2^8, 4)   # flow field size with μ₀ concatenated
    output_dim = (2^8, 2^8, 2)  # size of reconstructed RHS field
    conv_kernel = 3             # DO NOT CHANGE
    pool_kernel = 4             # DO NOT CHANGE  
    n_conv = 4                  # number of convolutional layers
    n_dense = 2                 # number of dense layers in bottleneck
    latent_dim = 16             # latent dimension
    stride = 1                  # stride for convolutions
    C_base = 16                 # first amount of channels for convs
    normalize = true            # normalise training data
    save_path = "data/Lux_models"   # results path
    data_path = "data/datasets/RE2500/2e8/U_128_period.jld2"
    full_data_path = "data/datasets/RE2500/2e8/U_128_full.jld2"
end

function get_data_in(X, μ₀; idx=nothing)
    if isnothing(idx)
        Xin = cat(X, μ₀; dims=3)
    else
        Xtarget = X[:, :, :, idx]
        μbatch = μ₀[:, :, :, idx]
        Xin = cat(Xtarget, μbatch; dims=3)
    end
    return (Xin, Xtarget, μbatch)
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


function get_data(batch_size, path;t_training=10, n_training=500, n_test=500, split=0.2, verbose=true)

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

    test_idx = downsample_equal(collect(last(train_idx_combined)+1 : N), n_test)

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

# Helper functions for parametric construction
function enc_layer(k, p, Cin, Cout, stride; BN=true)
    if BN
        return Chain(
            Conv((k, k), Cin => Cout, identity; pad=SamePad(), stride=stride, cross_correlation=true),
            BatchNorm(Cout),
            relu,
            MaxPool((p, p))
        )
    else
        return Chain(
            Conv((k, k), Cin => Cout, identity; pad=SamePad(), stride=stride, cross_correlation=true),
            relu,
            MaxPool((p, p))
        )
    end
end


Encoder(args::LuxArgs; verbose=true) = Encoder_parametric(args.input_dim, args.latent_dim;
                            n_conv=args.n_conv, n_dense=args.n_dense, C_base=args.C_base,
                            conv_kernel=args.conv_kernel, pool_kernel=args.pool_kernel,
                            stride=args.stride, verbose=verbose)

# Parametric encoder constructor
function Encoder_parametric(input_size::Tuple{Int,Int,Int}, latent_dim::Int;
                           n_conv=4, n_dense=3, C_base=8, conv_kernel=3,
                           pool_kernel=2, stride=1, verbose=true)
    H, W, C_in = input_size
    # Build convolutional layers
    enc_layers = []
    enc_channels = []
    
    push!(enc_layers, enc_layer(conv_kernel, pool_kernel, C_in, C_base, stride))
    push!(enc_channels, (C_in, C_base))
    
    for i in 1:(n_conv - 1)
        C1 = C_base * 2^(i - 1)
        C2 = C_base * 2^i
        
        push!(enc_channels, (C1, C2))
        is_last = (i == n_conv - 1)  # this is the last conv block
        push!(enc_layers,
              enc_layer(conv_kernel, pool_kernel, C1, C2, stride; BN = !is_last))        
    end
    
    # Calculate output size after convolutions
    rng = Xoshiro(0)
    dummy = rand(Float32, H, W, C_in, 2)
    temp_conv = Chain(enc_layers...)
    temp_p, temp_st = Lux.setup(rng, temp_conv)
    temp_st = LuxCore.testmode(temp_st)
    temp_out, _ = temp_conv(dummy, temp_p, temp_st)
    H_temp, B_temp, C_temp ,_ = size(temp_out)
    dense_in = H_temp * B_temp * C_temp
    
    # Build dense layers
    dense_layers = Any[FlattenLayer()]
    dense_nodes = []
    
    for k in 0:(n_dense - 2)
        # nodes = Int.(pool_kernel .^ (log(pool_kernel, dense_in)))
        # nodes = Int.(2 .^ (log2(dense_in) .- 2 .* (k, k+1)))
        nodes = Int.(dense_in .* 1 ./ (2^k, 2^(k+1)))
        if nodes[end] ≤ latent_dim
            @warn "Requested amount of dense layers to high for "
            break
        end
        push!(dense_nodes, nodes)
        push!(dense_layers, Dense(nodes...))
        push!(dense_layers, relu)
    end
    
    final_nodes = (isempty(dense_nodes) ? dense_in : dense_nodes[end][end], latent_dim)
    push!(dense_nodes, final_nodes)
    push!(dense_layers, Dense(final_nodes...))
    
    if verbose
        enc_channel_str = join(["$(c[1])→$(c[2])" for c in enc_channels], " -> ")
        dense_str = join(["$(c[1])→$(c[2])" for c in dense_nodes], " -> ")

        dims = [input_size]
        for j in 1:length(enc_layers)
            sub = Chain(enc_layers[1:j]...)
            p, st = Lux.setup(rng, sub)
            st = LuxCore.testmode(st)
            out, st = sub(dummy, p, st)
            push!(dims, size(out)[1:3])
        end
        dims_str = join(["$dim" for dim in dims], " -> ")
        @info "Parametric Encoder built:"
        @info "  Conv: $enc_channel_str"
        @info "  Sizes: $dims_str"
        @info "  Dense: $dense_str"
    end
    layers = Chain(enc_layers..., dense_layers...)
    return Encoder(layers)
end
    


# Encoder(input_size::Tuple{Int,Int,Int}, latent_dim::Int; hidden_dim=256, C_next::Int=4, padding=1, stride=2, verbose::Bool=true) = begin
#     H, W, C = input_size

#     convpart = Chain(
#         Conv((3, 3), C => C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
#         MaxPool((2, 2)),
#         Conv((3, 3), C_next => 2C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
#         MaxPool((2, 2)),
#         Conv((3, 3), 2C_next => 4C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
#         MaxPool((2, 2)),
#         Conv((3, 3), 4C_next => 8C_next, identity; pad=padding, stride=stride, cross_correlation=true), relu,
#         MaxPool((2, 2)),
#         FlattenLayer()  # instantiate
#         # more pooling and Conv
#     )
#     dummy = zeros(Float32, H, W, C, 1)

#     # initialize convpart params/state and run forward to infer flattened size
#     rng = Xoshiro(0)
#     ps_conv, st_conv = Lux.setup(rng, convpart)
#     flat, st_conv = convpart(dummy, ps_conv, st_conv)
#     dense_in = size(flat, 1)

#     verbose && @info "Initialize Encoder with $(dense_in) → $(hidden_dim) → $(latent_dim) bottleneck"

#     layers = Chain(
#         convpart,
#         Dense(dense_in, hidden_dim), gelu,
#         Dense(hidden_dim, latent_dim)      # latent_dim = 16 later
#     )

#     return Encoder(layers)
# end

function (encoder::Encoder)(x, ps, st)
    z, st_new = encoder.layers(x, ps, st)
    return z, st_new
end

struct Decoder{L} <: AbstractLuxWrapperLayer{:layers}
    layers::L
end


function upsample2(x, p)
    H, W, _, _ = size(x)
    return NNlib.upsample_bilinear(x; size=(p*H, p*W))
end

function dec_layer(k, p, Cin, Cout, stride)
    return Chain(
        x -> upsample2(x, p),
        Conv((k, k), Cin => Cout; pad=SamePad(), stride=stride),
        # BatchNorm(Cout),
        gelu
    )
end

Decoder(args::LuxArgs; verbose=true) = Decoder_parametric(args.output_dim, args.latent_dim;
                         n_conv=args.n_conv, n_dense=args.n_dense, C_base=args.C_base,
                        conv_kernel=args.conv_kernel, pool_kernel=args.pool_kernel,
                        stride=args.stride, verbose=verbose)


function construct_dense_nodes(n_dense::Int, latent_dim::Int, dense_max::Int)
    dense_nodes = []
     for k in 0:(n_dense - 2)
        nodes = Int.(dense_max .* 1 ./ (2^k, 2^(k+1)))
        if nodes[end] ≤ latent_dim
            @warn "Amount of requested dense layers to high, stopping early"
            break
        end
        push!(dense_nodes, nodes)
    end
    return dense_nodes
end

function Decoder_parametric(output_size::Tuple{Int,Int,Int}, latent_dim::Int;
                           n_conv=4, n_dense=3, C_base=8, conv_kernel=3,
                           pool_kernel=2, stride=1, verbose=true)
    H, W, C_out = output_size
    
    # Calculate compression ratio from encoder
    # This should match the encoder's final spatial size
    cr = pool_kernel^n_conv
    h_lat, w_lat = div(H, cr), div(W, cr)
    channels_mid = C_base * 2^(n_conv - 1)
    
    # Build dense layers (reverse of encoder)
    dense_layers = []
    dense_out = h_lat * w_lat * channels_mid
    
    # Calculate dense layer sizes
    dense_nodes = []
     for k in 0:(n_dense - 2)
        nodes = Int.(dense_out .* 1 ./ (2^k, 2^(k+1)))
        if nodes[end] ≤ latent_dim
            break
        end
        push!(dense_nodes, nodes)
    end

    final_nodes = (isempty(dense_nodes) ? dense_out : dense_nodes[end][end], latent_dim)
    push!(dense_nodes, final_nodes)
    dense_nodes = reverse(reverse.(dense_nodes))

    
    # Build dense chain
    for (i, (n_in, n_out)) in enumerate(dense_nodes)
        push!(dense_layers, Dense(n_in, n_out))
        if i < length(dense_nodes)
            push!(dense_layers, relu)
        end
    end
    
    # Reshape layer
    reshape_layer = x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2))
    rng = Xoshiro(0)
    latent = zeros(Float32, h_lat, w_lat, channels_mid, 2)

    # Build deconvolutional layers
    dec_layers = Any[reshape_layer]
    dec_channels = []
    
    C1_dec = C_base * 2^(n_conv - 1)
    C2_dec = C_base * 2^(n_conv - 2)
    
    for i in 1:(n_conv)
        push!(dec_channels, (C1_dec, C2_dec))
        push!(dec_layers, dec_layer(conv_kernel, pool_kernel, C1_dec, C2_dec, stride))
        C1_dec, C2_dec = C2_dec, max(1, Int(C2_dec ÷ 2))
    end
    
    # Final output layer
    C_last = dec_channels[end][end]
    push!(dec_channels, (C_last, C_out))
    push!(dec_layers,
            # Chain( x -> upsample2(x, pool_kernel), 
            Conv((conv_kernel, conv_kernel), C_last => C_out; pad=SamePad(), stride=stride)
            )
    
    # Smoothing layer
    push!(dec_channels, (C_out, C_out))
    push!(dec_layers, Conv((conv_kernel, conv_kernel), C_out => C_out; pad=SamePad(), stride=stride))
    if verbose
        dec_channel_str = join(["$(c[1])→$(c[2])" for c in dec_channels], " -> ")
        dense_str = join(["$(c[1])→$(c[2])" for c in dense_nodes], " -> ")
        dims = [size(latent)[1:3]]
        for j in 2:length(dec_layers)
            sub = Chain(dec_layers[2:j]...)
            p, st = Lux.setup(rng, sub)
            st = LuxCore.testmode(st)
            out, _ = sub(latent, p, st)
            push!(dims, size(out)[1:3])
        end
        dims_str = join(["$dim" for dim in dims], " -> ")

        @info "Parametric Decoder built:"
        @info "  Dense: $dense_str"
        @info "  Sizes: $dims_str"
        @info "  Conv: $dec_channel_str"
    end
    
    layers = Chain(dense_layers..., dec_layers...)
    return Decoder(layers)
end

# Decoder(output_size::Tuple{Int,Int,Int}, latent_dim::Int; hidden_dim=256, C_next::Int=4, verbose::Bool=true) = begin
#     H, W, C = output_size
#     h_lat, w_lat = div(H, 16), div(W, 16)
#     channels_mid = 8 * C_next
#     dense_len = h_lat * w_lat * channels_mid

#     layers = Chain(
#         Dense(latent_dim, hidden_dim), gelu,
#         Dense(hidden_dim, dense_len),
#         x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2)),

#         # stage 1: H/16 → H/8
#         x -> upsample2(x),
#         Conv((3, 3), 8C_next => 4C_next; pad=1), gelu,

#         # stage 2: H/8 → H/4
#         x -> upsample2(x),
#         Conv((3, 3), 4C_next => 2C_next; pad=1), gelu,

#         # stage 3: H/4 → H/2
#         x -> upsample2(x),
#         Conv((3, 3), 2C_next => 1C_next; pad=1), gelu,

#         # stage 4: H/2 → H
#         x -> upsample2(x),
#         Conv((3, 3), C_next => C; pad=1),     # linear output (no relu!)
#         Conv((3, 3), C => C; pad=1)           # small anti-alias / smoothing
#     )
#     verbose && @info "Initialize Decoder (upsample+conv) with $(latent_dim) → $(hidden_dim) → $(dense_len)"

#     return Decoder(layers)
# end

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
# function total_loss(m::AE, ps, st, 
#     x_in::AbstractArray,        # (u,v,mask)
#     x_target::AbstractArray,    # (u,v)
#     μ₀::AbstractArray;          # (H,W,1,B)  1 outside / 0 inside
#     loss=:L1,
#     λdiv=0f0,
#     λmask=0f0)

#     st2 = st
#     Lrec = 0f0
#     Linside = 0f0
#     L2div = 0f0
#     corrs = Float32[]

#     # 1) Forward pass through AE (Lux-style)
#     Zygote.ignore() do
#         @timeit Main.to "AE forward" begin
#             # do not put heavy compute inside ignore; only the timing wrapper
#         end
#     end
#     x̂, st2 = m(x_in, ps, st)

#     # correlations
#     Zygote.ignore() do
#         @timeit Main.to "correlations" begin end
#     end
#     corrs = batch_corrs(x_target, x̂)

#     # apply mask
#     Zygote.ignore() do
#         @timeit Main.to "apply mask" begin end
#     end
#     x̂ = x̂ .* μ₀

#     if λmask != 0f0
#         Zygote.ignore() do
#             @timeit Main.to "masked_loss" begin end
#         end
#         Lrec, Linside = masked_loss(x_target, x̂, μ₀; loss=loss)
#     elseif λdiv != 0f0
#         Zygote.ignore() do
#             @timeit Main.to "divergence_loss" begin end
#         end
#         try
#             L2div = div_loss_L2(x̂)
#         catch e
#             @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
#         end
#         Zygote.ignore() do
#             @timeit Main.to "recon_loss" begin end
#         end
#         Lrec = recon_loss(x_target, x̂; loss=loss)
#     else
#         Zygote.ignore() do
#             @timeit Main.to "recon_loss" begin end
#         end
#         Lrec = recon_loss(x_target, x̂; loss=loss)
#     end

#     # aggregate
#     Zygote.ignore() do
#         @timeit Main.to "total_loss aggregate" begin end
#     end
#     L = Lrec + λdiv * L2div + λmask * Linside
#     return L, st2, (Lrec, Linside, L2div, corrs)
# end

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

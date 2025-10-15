using JLD2
using CUDA, cuDNN
using Flux
using Flux: glorot_uniform, Conv, ConvTranspose, Dense, Chain, relu
using Optimisers: AdamW
using WaterLily
using Random
using Statistics
using ProgressMeter: Progress, next!
using Todo
using MLUtils: DataLoader
using Zygote


includet("../custom.jl")


function get_data(batch_size; tmin=-1, tmax=-1, n_samples=500)
    @load "/home/matth/Thesis/data/RHS_shedding_data_arr.jld2" RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=n_samples)
    X = cat(RHS_data["RHS"]...; dims=4)   # now X has shape (H,W,C,N)
    X = Float32.(X)
    return DataLoader(X, batchsize=batch_size, shuffle=true)
end


struct Encoder 
    layers::Chain
end


Flux.@layer Encoder

Encoder(in_channels::Int, latent_dim::Int; C_next::Int=4) = Encoder(Chain(
    Conv((3,3), in_channels => C_next, relu; pad=(1,1), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), C_next => 2*C_next, relu; pad=(1,1), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), 2*C_next => 4*C_next, relu; pad=(1,1), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), 4*C_next => 8*C_next, relu; pad=(1,1), stride=(1,1)),
    MaxPool((2,2)),
    Flux.flatten,
    # x -> reshape(x, :, size(x, 4)),  # flatten not use parametric layers, optimiser does not like it
    # x -> Dense(size(x, 1), latent_dim)(x)  # Parametric Dense layer
))


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


Decoder(latent_dim::Int, out_channels::Int; C_next::Int=4) = Decoder(Chain(
    # z : (n_latent, T) -> (24*8*(8*C_next), T)
    # x -> Dense(latent_dim, 24 * 8 * (8 * C_next))(x),   # <<< explicit multiplication
    LazyDense(24 * 8 * (8 * C_next)),    # produces (features, batch)
    # -> (24, 8, 8*C_next, T)
    x -> reshape(x, 24, 8, 8*C_next, size(x, 2)),
    ConvTranspose((2, 2), 8*C_next => 4*C_next, relu; stride=(2, 2), pad=(0, 0)),
    ConvTranspose((2, 2), 4*C_next => 2*C_next, relu; stride=(2, 2), pad=(0, 0)),
    ConvTranspose((2, 2), 2*C_next => C_next,   relu; stride=(2, 2), pad=(0, 0)),
    ConvTranspose((2, 2), C_next   => out_channels,       relu; stride=(2, 2), pad=(0, 0)),
    ConvTranspose((3, 3), out_channels => out_channels, identity; stride=(1, 1), pad=(0, 0))
))
todo"check architecture of decoder"

function reconstruct(encoder::Encoder, decoder::Decoder, x)
    z = encoder(x)
    return decoder(z)
end

# function divergence_field(u; mean=false, max=false)
#     println(typeof(u))
#     H, W, _ = size(u)
#     σ = zeros(eltype(u), H, W)
#     @inside σ[I] = WaterLily.div(I, u)
#     println(size(u), size(σ))
#     if mean
#         return mean(σ)
#     elseif max
#         return maximum(σ)
#     else
#         return σ
#     end
# end
function divergence_field(u; mean=false, max=false)
    if ndims(u) == 4
        H, W, C, N = size(u)
        σ = zeros(eltype(u), H, W, N)
        # @info "divergence_field: input is batched (H,W,C,N) = $(size(u)), computing per-sample"
        for n in 1:N
            # compute divergence for single sample (H,W,C)
            σ[:, :, n] = divergence_field(view(u, :, :, :, n); mean=false, max=false)
        end
        if mean
            return mean(σ)
        elseif max
            return maximum(σ)
        else
            return σ
        end
    elseif ndims(u) == 3
        H, W, C = size(u)
        σ = zeros(eltype(u), H, W)
        @inside σ[I] = WaterLily.div(I, u)
        # @info "divergence_field: single sample sizes (u, σ) = $(size(u)), $(size(σ))"
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
mae_loss(ŷ,x) = mean(abs, ŷ .- x)                    # optional L1

Zygote.@nograd divergence_field
Zygote.@nograd div_loss_L2

# combined total loss (ŷ = decoder(z) or ae(x))
function total_loss(encoder, decoder, x; λdiv=1)
    ŷ = reconstruct(encoder, decoder, x)
    Lrec = recon_loss(ŷ, x)
    L2div = zero(eltype(Lrec))
    if λdiv != 0
        try
            # If divergence computation fails on GPU we skip it (avoid CPU/GPU mix)
            L2div = div_loss_L2(ŷ)
        catch e
            @warn "div_loss_L2 failed (likely GPU/CPU mismatch). skipping divergence loss: $e"
            L2div = zero(eltype(Lrec))
        end
    end
    return Lrec + λdiv * L2div, (Lrec, L2div)
end

Base.@kwdef mutable struct Args
    η = 1e-3                # learning rate
    λ = 1e-4                # regularization paramater
    batch_size = 10        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 2             # number of epochs
    seed = 42                # random seed
    use_gpu = false              # use GPU
    input_dim = (386, 130, 2)        # flow field size
    latent_dim = 64          # latent dimension
    verbose_freq = 5       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

function train(; kws...)

    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.use_gpu
        device = Flux.get_device()
    else
        device = Flux.get_device("CPU")
    end

    @info "Training on $device"

    # load RHS data
    loader = get_data(args.batch_size)

    _, _, C = args.input_dim

    # initialize encoder and decoder
    encoder = Encoder(C, args.latent_dim) |> device
    decoder = Decoder(args.latent_dim, C) |> device

    # define optimizer
    opt_enc = Flux.setup(AdamW(eta=args.η, lambda=args.λ), encoder)
    opt_dec = Flux.setup(AdamW(eta=args.η, lambda=args.λ), decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # training
    @info "Start Training, total $(args.epochs) epochs"

    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))
        for x in loader 
            x_dev = x |> device

            @info "batch: eltype=$(eltype(x_dev)) typeof=$(typeof(x_dev)) size=$(size(x_dev))"
            
            loss, (grad_enc, grad_dec) = Flux.withgradient(encoder, decoder) do enc, dec
                    total_loss(enc, dec, x_dev)
            end
        
            Flux.update!(opt_enc, encoder, grad_enc)
            Flux.update!(opt_dec, decoder, grad_dec)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 
        end
    end
end

train()
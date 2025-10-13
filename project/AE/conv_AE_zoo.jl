using JLD2
using MLUtils: DataLoader
includet("../custom.jl")


function get_data(batch_size; tmin=-1, tmax=-1, n_samples=500)
    @load "/home/matth/Thesis/data/RHS_shedding_data_arr.jld2" RHS_data
    downsample_RHS_data!(RHS_data; tmin=tmin, tmax=tmax, n_samples=n_samples)
    X = cat(RHS_data["RHS"]...; dims=4)   # now X has shape (H,W,C,N)
    return DataLoader(X, batchsize=batch_size, shuffle=true)
end


struct Encoder 
    layers::Chain
end


Flux.@layer Encoder

Encoder(in_channels::Int, latent_dim::Int) = Encoder(Chain(
    Conv((3,3), in_channels => 4, relu; pad=(1,1), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), 4 => 8, relu; pad=(2,2), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), 8 => 16, relu; pad=(3,3), stride=(1,1)),
    MaxPool((2,2)),
    Conv((3,3), 16 => 32, relu; pad=(4,4), stride=(1,1)),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),  # flatten
    Dense(:, latent_dim)              # parametric layer to latent space
))

function (encoder::Encoder)(x)
    z = encoder.layers(x)
    return z
end


struct Decoder 
    layers::Chain
end

Flux.@layer Decoder


Decoder(latent_dim::Int, out_channels::Int) = Decoder(Chain(
    # z : (n_latent, T) -> (24*8*8C_next, T)
    x -> Dense(n_latent, 24 * 8 * (8C_next))(x),
    # -> (24, 8, 8C_next, T)
    x -> reshape(x, 24, 8, 8C_next, size(x, 2)),
    # Upsample back to original spatial size via stride-2 deconvs
    # 24×8×(8C_next) -> 48×16×(4C_next)
    ConvTranspose((2, 2), 8C_next => 4C_next, relu; stride=(2, 2), pad=(0, 0)),
    # 48×16×(4C_next) -> 96×32×(2C_next)
    ConvTranspose((2, 2), 4C_next => 2C_next, relu; stride=(2, 2), pad=(0, 0)),
    # 96×32×(2C_next) -> 192×64×(C_next)
    ConvTranspose((2, 2), 2C_next => C_next,   relu; stride=(2, 2), pad=(0, 0)),
    # 192×64×(C_next) -> 384×128×C  (C is original input channels, here 2)
    ConvTranspose((2, 2), C_next   => C,       relu; stride=(2, 2), pad=(0, 0)),

    # Now 384×128×C; add +2 in both H and W to match 386×130
    # Keep identity activation for reconstruction
    ConvTranspose((3, 3), C => C, identity; stride=(1, 1), pad=(0, 0))
))


function reconstruct(encoder::Encoder, decoder::Decoder, x)
    z = encoder(x)
    return decoder(z)
end

function divergence_field(u) # u::Array{T,4} => (H,W,C,N)
    H,W,C,N = size(u)
    σ = zeros(eltype(u), H, W, N)
    for n in 1:N
        un = view(u, :, :, :, n)                # (H,W,C) for this sample
        for I in CartesianIndices((H,W))
            σ[I[1], I[2], n] = div(I, un)       # div from Flow.jl expects (H,W,C)
        end
    end
    return σ
end

# losses
recon_loss(ŷ, x) = mean(abs2, ŷ .- x)            # MSE
div_loss(u) = mean(abs2, divergence_field(u))     # L2 of divergence field
mae_loss(ŷ,x) = mean(abs, ŷ .- x)                # optional L1

# combined total loss (ŷ = decoder(z) or ae(x))
function total_loss(ae, x; λdiv=1e-3, λgrad=0.0)
    ŷ = ae(x)
    Lrec = recon_loss(ŷ, x)
    Ldiv = div_loss(ŷ)           # penalize divergence of reconstruction
    # optional gradient term could be added here
    return Lrec + λdiv * Ldiv, (Lrec, Ldiv)
end
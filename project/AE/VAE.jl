module VAE

using Flux
using Flux: @epochs, mse, glorot_uniform, params
using Statistics

# Define the encoder network
struct Encoder
    net::Chain
    μ_layer::Dense
    logvar_layer::Dense
end

function Encoder(input_dim::Int, latent_dim::Int, hidden_dims::Vector{Int})
    layers = []
    in_dim = input_dim
    for h in hidden_dims
        push!(layers, Dense(in_dim, h, relu; init=glorot_uniform))
        in_dim = h
    end
    net = Chain(layers...)
    μ_layer = Dense(in_dim, latent_dim)
    logvar_layer = Dense(in_dim, latent_dim)
    return Encoder(net, μ_layer, logvar_layer)
end

function (enc::Encoder)(x)
    h = enc.net(x)
    μ = enc.μ_layer(h)
    logvar = enc.logvar_layer(h)
    return μ, logvar
end

# Define the decoder network
struct Decoder
    net::Chain
end

function Decoder(latent_dim::Int, output_dim::Int, hidden_dims::Vector{Int})
    layers = []
    in_dim = latent_dim
    for h in hidden_dims
        push!(layers, Dense(in_dim, h, relu; init=glorot_uniform))
        in_dim = h
    end
    push!(layers, Dense(in_dim, output_dim))
    net = Chain(layers...)
    return Decoder(net)
end

function (dec::Decoder)(z)
    return dec.net(z)
end

# Reparameterization trick
function reparameterize(μ, logvar)
    std = exp.(0.5 .* logvar)
    ϵ = randn(size(std))
    return μ .+ std .* ϵ
end

# VAE model
struct VAEModel
    encoder::Encoder
    decoder::Decoder
end

function VAEModel(input_dim::Int, latent_dim::Int, hidden_dims::Vector{Int})
    encoder = Encoder(input_dim, latent_dim, hidden_dims)
    decoder = Decoder(latent_dim, input_dim, reverse(hidden_dims))
    return VAEModel(encoder, decoder)
end

# Forward pass: encode, sample, decode
function (vae::VAEModel)(x)
    μ, logvar = vae.encoder(x)
    z = reparameterize(μ, logvar)
    x̂ = vae.decoder(z)
    return x̂, μ, logvar
end

# Loss function: reconstruction + KL divergence
function vae_loss(vae::VAEModel, x)
    x̂, μ, logvar = vae(x)
    rec_loss = mse(x̂, x)
    kl_loss = -0.5 * sum(1 .+ logvar .- μ.^2 .- exp.(logvar)) / size(x, 2)
    return rec_loss + kl_loss, rec_loss, kl_loss
end

# Compress: encode to latent space
function compress(vae::VAEModel, x)
    μ, _ = vae.encoder(x)
    return μ
end

# Decompress: decode from latent space
function decompress(vae::VAEModel, z)
    return vae.decoder(z)
end

export VAEModel, vae_loss, compress, decompress

end # 

# Example training loop for the VAE
using JLD2
@load "matth/data/RHS_shedding_data_arr.jld2" RHS_data
RHS_data["flattened"] = [vec(r) for r in RHS_data["RHS"]]

println(size(RHS_data["flattened"][1])[1])

first = VAE.VAEModel(size(RHS_data["flattened"][1])[1], 20, [2^5, 2^4])


# Example usage:
# input_dim = size of your input data (number of features)
# latent_dim = desired latent space dimension
# hidden_dims = vector of hidden layer sizes, e.g., [128, 64]
# vae = VAEModel(input_dim, latent_dim, hidden_dims)

function train_vae!(vae, data, epochs=10, lr=1e-3)
    opt = Adam(lr)
    ps = params(vae)
    for epoch in 1:epochs
        total_loss = 0.0
        for x in data
            loss, _, _ = vae_loss(vae, x)
            Flux.Optimise.update!(opt, ps, loss)
            total_loss += loss
        end
        println("Epoch $epoch, Loss: $(total_loss / length(data))")
    end
end

export train_vae!

train_vae!(first, RHS_data["flattened"], 10, 1e-3)


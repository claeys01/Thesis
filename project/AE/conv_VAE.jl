using Flux
using Flux: glorot_uniform
using JLD2

includet("../custom.jl")


@load "/home/matth/Thesis/data/RHS_shedding_data_arr.jld2" RHS_data
downsample_RHS_data!(RHS_data; tmin=50, tmax=75, n_samples=50)
snapshot = cat(RHS_data["RHS"][1:10]...; dims=4)  # shape: (386, 130, 2, 10)
println("Input snapshot size: ", size(snapshot))

H, W, C, T = size(snapshot)

# encoder parameters
n_latent = 128  # Latent space dimension
C_next = 4
kernel_size = (3, 3)
padding=1

# Convolutional encoder
kernel1 = (3, 3)
kernel2 = (5, 5)
kernel3 = (7, 7)
kernel4 = (9, 9)

pooling_encoder = Chain(
    Conv(kernel1, C => C_next, relu; pad=(1, 1)),       
    MaxPool((2, 2)),                                     
    Conv(kernel2, C_next => 2C_next, relu; pad=(1, 1)), 
    MaxPool((2, 2)),
    Conv(kernel3, 2C_next => 4C_next, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    Conv(kernel4, 4C_next => 8C_next, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),  # Flatten
)


mid = Chain(
    x -> Dense(size(x, 1), n_latent)(x)  # Parametric Dense layer
)

decoder = Chain(
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
)


ae = Chain(pooling_encoder, mid, decoder)


output = pooling_encoder(snapshot)
println("Output size: ", size(output))

CR = (386 * 130 * 2) / size(output, 1) 
println("Compression ratio: ", CR)



ae_output = ae(snapshot)
println("Autoencoder output size: ", size(ae_output))



# stride_encoder = Chain(
#     Conv(kernel1, C=>32,  stride=2, pad=1), relu,  # 386x130 -> 193x65
#     Conv(kernel2, 32=>64, stride=2, pad=1), relu,  # 97x33
#     Conv(kernel3, 64=>128,stride=2, pad=1), relu,  # 49x17
#     Conv(kernel4, 128=>256,stride=2, pad=1), relu, # 25x9
#     x -> reshape(x, :, size(x,4)),                # 25*9*256=57600
#     Dense(57600, latent_dim)
# ) |> dev

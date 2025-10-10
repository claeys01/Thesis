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
n_latent = 16  # Latent space dimension
C_next = 4
kernel_size = (3, 3)
padding=1

# Convolutional encoder
kernel1 = (3, 3)
kernel2 = (5, 5)
kernel3 = (7, 7)
kernel4 = (9, 9)

conv_layers = Chain(
    Conv(kernel1, C => C_next, relu; pad=(1, 1)),        # (386, 130, 2) -> (386, 130, 32)
    MaxPool((2, 2)),                                     # (386, 130, 32) -> (193, 65, 32)
    Conv(kernel2, C_next => 2C_next, relu; pad=(1, 1)),  # (193, 65, 32) -> (193, 65, 64)
    MaxPool((2, 2)),
    Conv(kernel3, 2C_next => 4C_next, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    Conv(kernel4, 4C_next => 8C_next, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),  # Flatten
)


output = conv_layers(snapshot)
println("Output size: ", size(output))

CR = (386 * 130 * 2) / size(output, 1) 
println("Compression ratio: ", CR)


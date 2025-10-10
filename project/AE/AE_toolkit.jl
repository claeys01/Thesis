using Flux
using Flux: glorot_uniform
using JLD2

includet("../custom.jl")


@load "matth/data/RHS_shedding_data_arr.jld2" RHS_data
snapshot = cat(RHS_data["RHS"][1:10]...; dims=4)  # shape: (386, 130, 2, 10)
println("Input snapshot size: ", size(snapshot))

H, W, C, T = size(snapshot)

n_latent = 16  # Latent space dimension
C_next = 32
kernel_size = (3, 3)
padding=1

encoder = Chain(
    Conv(kernel_size, C=>C_next, pad=padding, stride=1, init=glorot_uniform), x -> leakyrelu(x, 0.2),

)

# Run a single snapshot through the chain
snapshot = cat(RHS_data["RHS"][1:10]...; dims=4)  # shape: (386, 130, 2, 10)
println("Input snapshot size: ", size(snapshot))

output = encoder(snapshot)
H_out, W_out, C_out, T_out = size(output)
println("Output size encoder: ", size(output))

println("Compression ratio encoder: ", (H * W * C) / (H_out * W_out * C_out))
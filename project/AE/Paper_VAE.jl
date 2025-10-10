using Flux
using Flux: glorot_uniform
using JLD2

@load "matth/data/RHS_shedding_data_arr.jld2" RHS_data


# Select indices corresponding to time 50 to 75
time_indices = findall(t -> t ≥ 50 && t ≤ 75, RHS_data["time"])
selected_indices = time_indices

# Downsample to ~50 entries
n_samples = 50
downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
final_indices = selected_indices[downsampled_indices]

# Downsample all relevant entries in RHS_data
RHS_data["time"] = RHS_data["time"][final_indices]
RHS_data["Δt"] = RHS_data["Δt"][final_indices]
RHS_data["RHS"] = RHS_data["RHS"][final_indices]

println("Downsampled to ", length(RHS_data["time"]), " time steps.")
println("Input data size: ", size(RHS_data["RHS"][1]))


kernel_size = (5, 5)


# Residual block (3×3, stride=1)
# Residual block (like PyTorch's nn.Sequential + skip connection)
function resblock2d(ch)
    conv1 = Conv(kernel_size, ch=>ch, pad=padding, stride=1, init=glorot_uniform)
    conv2 = Conv(kernel_size, ch=>ch, pad=padding, stride=1, init=glorot_uniform)
    return x -> x .+ conv2(relu(conv1(x)))
end

# Stack N residual blocks (like nn.Sequential of N blocks)
function resstack2d(ch, N=4)
    blocks = [resblock2d(ch) for _ in 1:N]
    return Chain(blocks...)
end


C_in  = 2     # (u,v) or add pressure -> 3
C_res = 4    # narrower than the paper’s spirit
padding = 2    

encoder2d_fast = Chain(
    # conv_1
    Conv(kernel_size, C_in=>C_res, pad=padding, stride=1, init=glorot_uniform), x -> leakyrelu(x, 0.2),

    # long skip over residual stack + conv_2
    # Residual stack with a skip connection (like PyTorch's nn.Sequential + skip)
    x -> begin
        skip = x  # Save input for skip connection
        # Pass through 4 residual blocks
        y = resstack2d(C_res, 12)(x)
        # Additional convolution layer
        y = Conv(kernel_size, C_res=>C_res, pad=padding, stride=1, init=glorot_uniform)(y)
        # Add skip connection (elementwise addition)
        skip .+ y
    end,

    # compress 0: stride-2 then expand x2
    Conv(kernel_size, C_res=>C_res, pad=padding, stride=2, init=glorot_uniform), x -> leakyrelu(x, 0.2),
    Conv(kernel_size, C_res=>2C_res, pad=padding, stride=1, init=glorot_uniform), x -> leakyrelu(x, 0.2),

    # compress 1
    Conv(kernel_size, 2C_res=>2C_res, pad=padding, stride=2, init=glorot_uniform), x -> leakyrelu(x, 0.2),
    Conv(kernel_size, 2C_res=>4C_res, pad=padding, stride=1, init=glorot_uniform), x -> leakyrelu(x, 0.2),

    # compress 2 (keep channels)
    Conv(kernel_size, 4C_res=>4C_res, pad=padding, stride=2, init=glorot_uniform), x -> leakyrelu(x, 0.2),
    Conv(kernel_size, 4C_res=>4C_res, pad=padding, stride=1, init=glorot_uniform), x -> leakyrelu(x, 0.2),

    # conv_out: halve channels (latent width)
    # Conv(kernel_size, 4C_res=>2C_res, pad=padding, stride=1, init=glorot_uniform),

    x -> reshape(x, :, size(x, 4)),  # Flatten

)
# Input shape: (H,W,C_in,N); output: (H/8, W/8, 2*C_res, N)

# Run a single snapshot through the chain
snapshot = cat(RHS_data["RHS"][1:10]...; dims=4)  # shape: (386, 130, 2, 10)
println("Input snapshot size: ", size(snapshot))

output_2 = encoder2d_fast(snapshot)
println("Output size encoder2d_fast: ", size(output_2))

println("Compression ratio encoder2d_fast: ", (386 * 130 * 2) / size(output_2, 1))


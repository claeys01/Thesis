using Lux
using NNlib
using Revise
using Random

includet("Lux_AE.jl")

args = LuxArgs()
rng = Xoshiro(0)

# # Define encoder layer: Conv -> ReLU -> MaxPool
# enc_layer(k, p, Cin, Cout, pad, stride) =
#     Chain(Conv((k, k), Cin => Cout, identity; pad, stride, cross_correlation=true),
#           relu,
#           MaxPool((p, p)))

# # Define decoder layer: Upsample -> Conv -> GELU
# dec_layer(k, p, Cin, Cout, pad) =
#     Chain(x -> NNlib.upsample_bilinear(x; size=(size(x,1)*p, size(x,2)*p)),
#           Conv((k, k), Cin => Cout; pad),
#           gelu)

# Setup input/output dimensions
size_pow = 8
input_dim = (2^size_pow, 2^size_pow, 4, 10)
output_dim = (2^size_pow, 2^size_pow, 2, 10)

x = rand(Float32, input_dim)
H, W, C_in, T = input_dim
_, _, C_out, _ = output_dim

# Architecture hyperparameters
n_conv = 6
n_dense = 3
C_base = 8
conv_kernel = 3
pool_kernel = 2
latent_dim = 16

# Build encoder convolutional layers
enc_layers = []
enc_channels = []

input_params = H*W*C_in
@info "Input dims: $(size(x)), input params: $(input_params)"

push!(enc_layers, enc_layer(conv_kernel, pool_kernel, C_in, C_base, args.stride))
push!(enc_channels, (C_in, C_base))

for i in 1:(n_conv - 1)
        C1 = C_base * 2^(i - 1)
        C2 = C_base * 2^i
        
        push!(enc_channels, (C1, C2))
        is_last = (i == n_conv - 1)  # this is the last conv block
        push!(enc_layers,
              enc_layer(conv_kernel, pool_kernel, C1, C2, stride; BN = !is_last))        
    end

# Test encoder and calculate compression ratio
temp_conv = Chain(enc_layers...)
temp_p, temp_st = Lux.setup(rng, temp_conv)
temp_out, _ = temp_conv(dummy, temp_p, temp_st)
H_temp, B_temp, C_temp ,_ = size(temp_out)
dense_in = H_temp*B_temp*C_temp

latent = vec(temp_out)
cr = div(input_params, dense_in)

# Track encoder layer outputs
for l in 1:length(enc_layers)
    sub = Chain(enc_layers[1:l]...)
    p, st = Lux.setup(rng, sub)
    out, st = sub(x, p, st)
    push!(enc_dof, length(vec(out)))
end

enc_channel_str = join(["$(c[1])→$(c[2])" for c in enc_channels], " -> ")
enc_param_string = join(["$i: $(p)" for (i, p) in enumerate(enc_dof)], " -> ")

@info "Encoder channel flow: $enc_channel_str"
@info "Encoder parameter flow: $enc_param_string, Conv CR: $cr"

# Build encoder dense layers
dense_in = enc_dof[end]
enc_dense_layers = []
enc_dense_nodes = []

for k in 0:(n_dense - 2)
    nodes = Int.(2 .^ (log2(dense_in) .- 2 .* (k, k+1)))
    if nodes[end] ≤ latent_dim
        break
    end
    push!(enc_dense_nodes, nodes)
    push!(enc_dense_layers, Dense(nodes...))
    push!(enc_dense_layers, gelu)
end

final_nodes = (enc_dense_nodes[end][end], latent_dim)
push!(enc_dense_nodes, final_nodes)
push!(enc_dense_layers, Dense(final_nodes...))

enc_dense_str = join(["$i, $(c[1])→$(c[2])" for (i, c) in enumerate(enc_dense_nodes)], " -> ")
@info "Encoder dense nodes flow: $enc_dense_str, Total CR: $(latent_dim/input_params))"

# Build decoder dense layers (reverse of encoder)
dec_dense_nodes = reverse(reverse.(enc_dense_nodes))

dec_dense_layers = reverse([
    layer isa Dense ?
        Dense(layer.out_dims => layer.in_dims, layer.activation) :
        layer
    for layer in enc_dense_layers
])

dec_dense_str = join(["$i, $(c[1])→$(c[2])" for (i, c) in enumerate(dec_dense_nodes)], " -> ")
@info "Decoder dense nodes flow: $dec_dense_str"

# Build decoder convolutional layers
h_lat, w_lat = div(H, cr), div(W, cr)
channels_mid = Int(C_base * cr ÷ 2)
@show h_lat, w_lat
dec_layers = Any[x -> reshape(x, h_lat, w_lat, channels_mid, size(x, 2))]
dec_channels = []

C1_dec = C_base * 2^(n_conv - 1)
C2_dec = C_base * 2^(n_conv - 2)

for _ in 1:(n_conv)
    push!(dec_channels, (C1_dec, C2_dec))
    push!(dec_layers, dec_layer(conv_kernel, pool_kernel, C1_dec, C2_dec, args.padding))
    C1_dec, C2_dec = C2_dec, Int(C2_dec ÷ 2)
end
#downsample full reconstruction to 2 channels
push!(dec_channels, (C_in, C_out))
push!(dec_layers, Conv((conv_kernel, conv_kernel), C_in => C_out; pad=args.padding))

# smoothing anti aliasing layer
push!(dec_channels, (C_out, C_out))
push!(dec_layers, Conv((conv_kernel, conv_kernel), C_out => C_out; pad=args.padding))

# Test decoder and track outputs
dec = Chain(dec_layers...)
dec_p, dec_st = Lux.setup(rng, dec)

dec_out, dec_st = dec(latent, dec_p, dec_st)
dec_channel_str = join(["$i, $(c[1])→$(c[2])" for (i, c) in enumerate(dec_channels)], " -> ")

dec_dof = []

for j in 1:length(dec_layers)
    sub = Chain(dec_layers[1:j]...)
    p, st = Lux.setup(rng, sub)
    out, st = sub(latent, p, st)
    push!(dec_dof, length(vec(out)))
end
@show size(dec_dof), size(dec_channels)

dec_param_string = join(["$i, $(p)" for (i, p) in enumerate(dec_dof)], " -> ")

@info "Decoder channel flow: $dec_channel_str"
@info "Decoder parameter flow: $dec_param_string"
@info "Output dims: $(size(dec_out)), output params: $(size(vec(dec_out))[end])"



using Lux
using NNlib
using Revise
using Random

includet("Lux_AE.jl")


# args = LuxArgs(n_conv=n_conv, n_dense=n_dense, C_base=C_base, 
#                 conv_kernel=conv_kernel, pool_kernel=pool_kernel, padding=padding)
# rng = Xoshiro(0)

args = LuxArgs()

enc = Encoder(args, verbose=true)
dec = Decoder(args, verbose=true)
ae = AE(enc, dec)
# nothing

# Hin, Win, Cin, T = 256, 256, 4, 100

# Cin = C
# Cout = 8

# dummy = rand(Float32, Hin, Win, Cin, T )

# ck = 5
# pk = 2

# # pad = H % ck
# pad = 2
# @show pad

# layer =  enc_layer(ck, pk, Cin, Cout, 1; BN=true)
# p, st = Lux.setup(rng, layer) 
# temp_out, _ = layer(dummy, p, st)
# @show size(dummy), size(temp_out)

# @show 256 % 6

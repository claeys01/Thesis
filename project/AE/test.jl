using Lux
using NNlib
using Revise
using Random

includet("Lux_AE.jl")

args = LuxArgs()
rng = Xoshiro(0)

# Architecture hyperparameters
# n_conv = 6
# n_dense = 3
# C_base = 8
# conv_kernel = 3
# pool_kernel = 2
# latent_dim = 16


enc = Encoder(args, verbose=true)
dec = Decoder(args, verbose=true)
ae = AE(enc, dec)
nothing
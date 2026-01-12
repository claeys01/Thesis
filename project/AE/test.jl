using Lux
using NNlib
using Revise
using Random
using JLD2
using Plots

includet("../utils/SimDataTypes.jl")
using .SimDataTypes
includet("Lux_AE.jl")

checkpoint = JLD2.load("data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2")
args_dict = checkpoint["args"]
@show args_dict
# args = LuxArgs(n_conv=n_conv, n_dense=n_dense, C_base=C_base, 
#                 conv_kernel=conv_kernel, pool_kernel=pool_kernel, padding=padding)
# rng = Xoshiro(0)

# a = range(1, 100, length=100)
# a = collect(1:10)
a = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
@show size(a)

idx = [1,3,5]
# b = downsample_equal(a, 7)
b= a[idx]
@show b

X = randn(256, 256, 2, 10)
μ = X

tup = get_data_in(X, μ; idx=idx)

TrainData = EpochData(tup...)
@show size(TrainData.Xin)
@show size(TrainData.Xin, 4)

args = LuxArgs()
nothing
data, loaders, normalizer = get_data(
            args.batch_size,
            args.full_data_path;
            n_training = args.train_downsample,
            n_test = args.test_downsample,
            split = 0.2,
            t_training = args.t_training,
            plotpath="jemoeder.png"
        )
# enc = Encoder(args, verbose=true)
# dec = Decoder(args, verbose=true)
# ae = AE(enc, dec)
nothing
# drop large objects and force a GC

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

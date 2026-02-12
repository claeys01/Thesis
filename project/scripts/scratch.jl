using Thesis
using Statistics
using Plots
using Revise

include("/home/matth/Thesis/project/scripts/data_getters/get_latent_data.jl")

checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
save_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2"
train_latent, test_latent = get_latent_data(checkpoint; save_path=save_path)

Thesis.train_NODE(NodeArgs(
    train_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_train.jld2",
    test_latent_path =  "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_test.jld2",
    total_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2"
))
# x = randn(256, 256, 2, 50)

# @show mean(x), std(x)
# x_norm, normalizer = normalize_batch(x, )
# @show mean(x_norm), std(x_norm)

# train_AE(LuxArgs())
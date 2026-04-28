using Thesis
using Statistics
using Plots
using CUDA
using Lux
using Random

# args = LuxArgs(
#     input_dim=(2^9, 2^9, 4),
#     output_dim=(2^9, 2^9, 2),
#     batch_size=16,
#     use_gpu=true,
# )

# device = get_device(; prefer_gpu=args.use_gpu)
#  @info "Using device: $device"

# enc = Encoder(args; verbose=true)
# dec = Decoder(args; verbose=true)
# ae = AE(enc, dec)

# rng = Xoshiro(args.seed)
# ps, st = Lux.setup(rng, ae)

# H, W, C = args.input_dim
# x = randn(Float32, H, W, C, args.batch_size)

# @info "Input size: $(size(x))"
# x̂, st = ae(x, LuxCore.testmode(ps), st)
# @info "Output size: $(size(x̂))"


checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
losses = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/loss_trajectory.jld2"

p = plot_losses(losses, checkpoint)
display(p)
savefig(p, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/loss_evolution.png")
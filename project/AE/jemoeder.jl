using JLD2
using Lux
using Random
using ProgressMeter
using Plots
using Dates
using Optimisers
using Enzyme
using Zygote
using DrWatson: struct2dict

includet("../custom.jl")
includet("Lux_AE.jl")
includet("../utils/Lux_AE_reconstructer.jl")
includet("../utils/Lux_AE_loss_plot.jl")

args = LuxArgs()
args.seed > 0 && Random.seed!(args.seed)

train_loader, validation_loader, normalizer = get_data(args.batch_size, args.full_data_path;
        n_samples=args.downsample, clip_bc=args.clip_bc, split=args.split, n_periods=args.n_periods)

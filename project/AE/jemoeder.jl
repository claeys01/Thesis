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

# simdata = load_simdata(args.full_data_path)
# # @show simdata.reordered_ranges

jemoeder = get_data(args.batch_size, args.full_data_path; n_training = 300, split=0.2, t_training=20.854)
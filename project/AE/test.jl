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

train_loader, _, normalizer = get_data(1000, "data/datasets/RE2500/2e8/U_128_full.jld2";
                                                        n_samples=1000, clip_bc=true, split=-1, field="u", verbose=false)

@show size(train_loader)

for batch in train_loader
    x_in, x_target, μ₀ = batch

    @show size(x_in)
end

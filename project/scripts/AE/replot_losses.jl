using Thesis
using Plots
using JLD2

# Pass a model directory (containing loss_trajectory.jld2 + checkpoint.jld2)
# as the first CLI arg, or edit the default below.
default_dir = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0"
model_dir = length(ARGS) >= 1 ? ARGS[1] : default_dir

loss_path = joinpath(model_dir, "loss_trajectory.jld2")
ckpt_path = joinpath(model_dir, "checkpoint.jld2")

isfile(loss_path) || error("missing loss_trajectory.jld2 in $model_dir")
isfile(ckpt_path) || error("missing checkpoint.jld2 in $model_dir")

p = plot_losses(loss_path, ckpt_path)
display(p)

out_path = joinpath(model_dir, "loss_evolution.png")
# savefig(p, out_path)
@info "saved" out_path

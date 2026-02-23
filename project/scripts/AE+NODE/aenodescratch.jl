using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs

reset_timer!(to::TimerOutput)


sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=true)


# load aenode struct with trained neural ai models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)

u, μ₀ = simdata.u, simdata.μ₀[:, :, :, 1]

# Compute Reynolds stress terms
uu, vv, uv = Thesis.RST(u, μ₀)
@show size(uu), size(vv), size(uv)

# Create contour plots using flood function


# Generate the plots
plt_rst, (plt_uu, plt_vv, plt_uv) = plot_reynolds_stresses(uu, vv, uv)
display(plt_rst)

GC.gc()
# Optionally save
# savefig(plt_rst, "figs/reynolds_stresses.png")





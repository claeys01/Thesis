using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs

reset_timer!(to::TimerOutput)

# # load aenode struct with trained neural models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)

u₀, μ₀ = simdata.u[:, :, :, 1], simdata.μ₀[:, :, :, 1]
u = simdata.u
simdata = nothing

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)
t_end = 50

sim.flow.u .= u₀
meanflow = MeanFlow(sim.flow; uu_stats=true)
while sim_time(sim) < t_end
        sim_step!(sim; remeasure=false, verbose=false)
        sim_info(sim)
        WaterLily.update!(meanflow, sim.flow)
end

# viz!(sim, ω_mean; clims=(-0.2,0.2), levels=60)
τ = uu(meanflow)
τ_uu = Array(τ[:, :, 1, 1])
@show size(τ_uu), typeof(τ_uu)
@show extrema(τ_uu)
# viz!(sim, τ_uu; clims=(-0.2,0.2), levels=60)
plt_rst1, _= Thesis.plot_reynolds_stresses(τ[:, :, 1, 1], τ[:, :, 2, 2], τ[:, :, 2, 1])
# display(plt_rst)

# Compute Reynolds stress terms
cuu, cvv, cuv = Thesis.RST(u, μ₀)
# @show size(uu), size(vv), size(uv)

# Generate the plots
plt_rst2, _ = Thesis.plot_reynolds_stresses(cuu, cvv, cuv)
# display(plt_rst)

plt = plot(plt_rst1, plt_rst2, layout=(2, 1))
display(plt)
nothing
# GC.gc()
# # Optionally save
# # savefig(plt_rst, "figs/reynolds_stresses.png")





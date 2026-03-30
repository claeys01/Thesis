using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs

reset_timer!(to::TimerOutput)

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false)

function pressure_plot(sim::BiotSimulation; title="")
    p_field = sim.flow.p .* sim.flow.μ₀[:, :, 1]
    clims = (-5, 5)
    flood(p_field, 
    # levels=20,
    # clims=clims,
    color=:viridis,
    aspectratio=:equal, 
    border=:none, 
    framestyle=:none,
    axis=nothing,
    plot_title= title,
    plot_titlefontsize=12,
    size = (400,350), 
    dpi=150,
    )
    
end

# # load aenode struct with trained neural models
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)

u₀, μ₀, t₀ = simdata.u[:, :, :, 1], simdata.μ₀[:, :, :, 1], simdata.time[1]

ref_forces = []

sim.flow.u .= copy(u₀)
sim_step!(sim, 5; verbose=true)
closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
push!(ref_forces, simdata.force[closest_idx])

plt1 = pressure_plot(sim; title="pressure at t₀")
u1_flood, _ = Thesis.velocity_flood(sim)
savefig(plot(u1_flood; plot_title="velocities at: t₀", titlefontsize=14), "figs/pressure_issue/u0.png")

savefig(plt1, "figs/pressure_issue/t0")
forces1 = Thesis.get_forces(sim)

t_end = 20
n_pred = 100
pred_Δt = 0.35f0
with_pred = true

# -------------------------------------- prediction ---------------------------------------------
predict_n!(sim, aenode, n_pred; Δt=pred_Δt, impose_biot=false)

closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
push!(ref_forces, simdata.force[closest_idx])
sim.flow.p .= simdata.p[:, :, closest_idx]

u_pred = copy(sim.flow.u)
diff_pred = mean(abs, u_pred[2:end-1, 2:end-1, :] .- simdata.u[2:end-1, 2:end-1, :, closest_idx])


plt2 = pressure_plot(sim; title="pressure at: t₀ + nₚ⋅Δt")
u2_flood, _ = Thesis.velocity_flood(sim)
savefig(plot(u2_flood; plot_title="velocities at: t₀ + nₚ⋅Δt", titlefontsize=14), "figs/pressure_issue/u2.png")

savefig(plt2, "figs/pressure_issue/t2")
forces2 = Thesis.get_forces(sim)

# -------------------------------------- sim_step 1 ---------------------------------------------
sim_step!(sim)

closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
push!(ref_forces, simdata.force[closest_idx])

u_sim1 = copy(sim.flow.u)

diff_u = mean(abs, u_pred[2:end-1, 2:end-1, :] .- u_sim1[2:end-1, 2:end-1, :])
@show mean(Thesis.div_vectorized(sim.flow.u))
@show "here"
plt3 = pressure_plot(sim; title="pressure at: t₀ + nₚ⋅Δt + CFL")
u3_flood, _ = Thesis.velocity_flood(sim)
savefig(plot(u3_flood; plot_title="velocities at: t₀ + nₚ⋅Δt + CFL", titlefontsize=14), "figs/pressure_issue/u3.png")

savefig(plt3, "figs/pressure_issue/t3")
forces3 = Thesis.get_forces(sim)

# -------------------------------------- sim_step 2 ---------------------------------------------


sim_step!(sim)
closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
push!(ref_forces, simdata.force[closest_idx])
plt4 = pressure_plot(sim; title="pressure at: t₀ + nₚ⋅Δt + 2⋅CFL")
u4_flood, _ = Thesis.velocity_flood(sim)
savefig(plot(u4_flood; plot_title="velocities at: t₀ + nₚ⋅Δt + 2⋅CFL", titlefontsize=14), "figs/pressure_issue/u4.png")
savefig(plt4, "figs/pressure_issue/t4")
forces4 = Thesis.get_forces(sim)

sim_step!(sim)
closest_idx = argmin(abs.(simdata.time .- sim_time(sim)))
push!(ref_forces, simdata.force[closest_idx])
plt5 = pressure_plot(sim; title="pressure at: t₀ + nₚ⋅Δt + 3⋅CFL")
u5_flood, _ = Thesis.velocity_flood(sim)
savefig(plot(u5_flood; plot_title="velocities at: t₀ + nₚ⋅Δt + 3⋅CFL", titlefontsize=14), "figs/pressure_issue/u5.png")
savefig(plt5, "figs/pressure_issue/t5")
forces5 = Thesis.get_forces(sim)

# u5_flood, _ = Thesis.velocity_flood(sim)
# display(plot(u5_flood; plot_title="velocities at: t₀ + nₚ⋅Δt + 3⋅CFL", titlefontsize=14))

# Plot forces with string x-axis
time_labels = ["t₀", "t₀ + nₚ⋅Δt", "t₀ + nₚ⋅Δt + CFL", "t₀ + nₚ⋅Δt + 2⋅CFL", "t₀ + nₚ⋅Δt + 3⋅CFL"]
drag_values = [forces1[1], forces2[1], forces3[1], forces4[1], forces5[1]]
lift_values = [forces1[2], forces2[2], forces3[2], forces4[2], forces5[2]]

ref_drag, ref_lift = first.(ref_forces), last.(ref_forces)
forces_plt = plot(
    1:5, drag_values,
    c=:red,
    xticks=(1:5, time_labels),
    label="Drag",
    ylabel="Force coefficient",
    title="Force Evolution",
    framestyle=:box,
    size=(700, 500),
    dpi=200,
    ylims = (-4,3),
    legend=:bottomright,
    margin=7Plots.mm,

)
plot!(forces_plt, 1:5, lift_values, label="Lift", c=:blue)
plot!(forces_plt, 1:2, drag_values[1:2], 
        label = "Prediction",
        color=:black, 
        lw=2, 
        marker=:circle, 
        markersize=2, 
        markerstrokewidth=1)

plot!(forces_plt, 1:2, lift_values[1:2], 
        label = "",
        color=:black, 
        lw=2, 
        marker=:circle, 
        markersize=2, 
        markerstrokewidth=1)

plot!(forces_plt, 1:5, ref_drag, ls=:dash, c=:red, label="Reference Drag")
plot!(forces_plt, 1:5, ref_lift, ls=:dash, c=:blue, label="Reference Lift")


vspan!(forces_plt, 1:2; fillcolor=:gray, alpha=0.2, label="Prediction region")
vspan!(forces_plt, 2:3; fillcolor=:blue, alpha=0.2, label="1 sim_step! after velocity injection")
vspan!(forces_plt, 3:4; fillcolor=:green, alpha=0.2, label="2 sim_step! after velocity injection")

savefig(forces_plt, "figs/pressure_issue/forces_closeup.png")
display(forces_plt)


GC.gc()
# combination_plt = plot(plt1, plt2, plt3, plt4, plt5; layout=(1, 5), size=(1600,350), dpi=500)
# nothing


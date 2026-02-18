using Thesis
using WaterLily
using Statistics
using Plots
using TimerOutputs

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=true)

reset_timer!(to::TimerOutput)

# load aenode struct with trained neural ai models
node_path = "data/NODE_models/Feb12-1551/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)


# random_int = findfirst(t -> t > aenode.ae_args.t_training, simdata.time)
random_int = 1
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)

train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_preds = []
n_pred_range = 1:12
for n_pred in n_pred_range
    t_pred = @elapsed Thesis.predict_n(aenode, u, μ₀, 2^n_pred, t₀; Δt=0f35, return_traj=false, impose_biot=false)
    @info "time needed for predicting $(2 ^ n_pred) steps: $t_pred"
    push!(t_preds, t_pred)
end
simdata = nothing
t_preds_ms = 1000 .* t_preds
n_steps = 2 .^ collect(n_pred_range)

plt = plot(n_steps, t_preds_ms;
    # Scale settings
    xscale = :log10,
    # yscale = :log10,
    
    # Labels
    xlabel = "Prediction Steps",
    ylabel = "Time (ms)",
    title = "AENODE Prediction Time vs Steps",
    
    # Line styling
    linewidth = 2,
    marker = :circle,
    markersize = 6,
    color = :royalblue,
    label = "Prediction time",
    
    # Grid and ticks
    grid = true,
    minorgrid = true,
    minorticks = 10,
    gridalpha = 0.3,
    minorgridalpha = 0.15,
    gridlinewidth = 0.5,
    
    # Tick formatting
    xticks = (n_steps, string.(n_steps)),  # Show actual step values
    
    # Font sizes
    guidefontsize = 11,
    tickfontsize = 9,
    titlefontsize = 10,
    legendfontsize = 9,
    
    # Frame and size
    framestyle = :box,
    size = (600, 400),
    dpi = 150,
    
    # Legend
    legend = :topleft,
    background_color_legend = RGBA(1, 1, 1, 0.8)
)

# Add a reference line showing linear scaling (optional)
# plot!(plt, n_steps, t_preds_ms[1] .* (n_steps ./ n_steps[1]);
#     linestyle = :dash,
#     color = :gray,
#     linewidth = 1.5,
#     label = "Linear scaling (reference)"
# )

plot!(plt, n_steps, n_steps .* 45.0;
    linestyle = :dash,
    color = :red,
    linewidth = 1.5,
    label = "WaterLily scaling"
)

display(plt)

# Save the plot
# savefig(plt, "figs/AENODE_trajectory_timescale.png")
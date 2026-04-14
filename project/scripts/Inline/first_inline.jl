ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ENV["THESIS_USE_CUDA"] = get(ENV, "THESIS_USE_CUDA", "true")

using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using Plots

# use_gpu = lowercase(get(ENV, "THESIS_USE_CUDA", "false")) == "true"

# t_inline = parse(Float32, get(ENV, "T_INLINE", "10.0"))
# t_train = parse(Float32, get(ENV, "T_TRAIN", "8.0"))
# t_accel_end = parse(Float32, get(ENV, "T_ACCEL_END", "25.0"))
# ae_epochs = parse(Int, get(ENV, "AE_EPOCHS", "1"))
# node_iters = parse(Int, get(ENV, "NODE_ITERS", "250"))
# n_switch = parse(Int, get(ENV, "N_SWITCH", "150"))
# pred_Δt = parse(Float32, get(ENV, "PRED_DT", "0.35"))
# save_interval = parse(Float32, get(ENV, "SAVE_INTERVAL", "0.5"))

Base.@kwdef struct InlineParams
    t_run = 10.0
    t_train = 8.0
    t_accel_end = 25
    ae_epochs = 1
    node_iters = 250
    n_switch = 150
    pred_Δt = 0.035
    save_interval = 1
end

params = InlineParams()

    
savedir = joinpath("data", "inline_runs", Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline.jld2")

u₀ = nothing
init_path = "data/datasets/RE2500/2e8/U_128_full.jld2"
initdata = load_simdata(init_path)
u₀ = copy(initdata.u[:, :, :, 1])
initdata = nothing; GC.gc()

sim = circle_shedding_biot(; mem=Array, perturb=false)
sim, simdata = run_sim(sim; t_end=t_inline, u₀=u₀, save_path=simdata_path, verbose=false)

ae_args = LuxArgs(
    simdata_ram=simdata,
    full_data_path=simdata_path,
    data_path=simdata_path,
    save_path=joinpath(savedir, "AE"),
    epochs=params.ae_epochs,
    t_training=params.t_train,
    use_gpu=is_hpc(),
)

ae, ae_ps, ae_st, AE_path = train_AE(ae_args; return_path=true)
normalizer = load_normalizer(AE_path)
ae_args.simdata_ram = nothing
# simdata = nothing; GC.gc()

node_args = NodeArgs(
    use_gpu=false,
    latent_dim=ae_args.latent_dim,
    maxiters=params.node_iters,
    downsample=ae_args.train_downsample,
    test_downsample=ae_args.test_downsample,
    extrapolate=false,
    save_path=joinpath(savedir, "NODE"),
)

node_path = train_NODE(node_args;
    ae=ae,
    ae_ps=ae_ps,
    ae_st=ae_st,
    normalizer=normalizer,
    ae_args=ae_args,
)

ae = nothing
normalizer = nothing
ae_ps = nothing
ae_st = nothing
GC.gc()

aenode = AENODE(AE_path, node_path)
plotdata = load_simdata(simdata_path)
train_idx, val_idx, test_idx = Thesis.get_idxs(plotdata, aenode.ae_args)
t_train_plot = plotdata.time[train_idx]
t_test_plot = plotdata.time[test_idx]
plotdata = nothing
GC.gc()

sim = circle_shedding_biot(; mem=Array, Re=Re, n=n, m=m, perturb=false)
if !isnothing(u₀)
    sim.flow.u .= u₀
end
sim_step!(sim)
ref_sim = deepcopy(sim)
sim_meanflow = MeanFlow(sim.flow; uu_stats=true)
ref_meanflow = MeanFlow(ref_sim.flow; uu_stats=true)

res = AccelResults()
n_integrs = Int[]
gif_frames = []
next_save = save_interval
step = 1

predict_flex(aenode, deepcopy(sim); Δt=pred_Δt, impose_biot=true)

while sim_time(sim) < t_accel_end
    if step % n_switch == 0 && sim_time(sim) > aenode.ae_args.t_training
        sim_time_before = sim_time(sim)
        predict_wall_time = @elapsed begin
            sim, n_integr = predict_flex(aenode, sim; Δt=pred_Δt, impose_biot=true, next_save=next_save)
        end
        sim_dt = sim_time(sim) - sim_time_before

        if n_integr != 0
            push!(n_integrs, n_integr)
            record_prediction!(res, sim, predict_wall_time, sim_dt, step)
        else
            wall_time = @elapsed sim_step!(sim)
            record_waterlily_step!(res, sim, wall_time)
        end
    else
        wall_time = @elapsed sim_step!(sim)
        record_waterlily_step!(res, sim, wall_time)
    end

    while sim_time(sim) > sim_time(ref_sim)
        step_reference!(res, ref_sim)
    end

    if sim_time(sim) > next_save
        WaterLily.update!(sim_meanflow, sim.flow)
        WaterLily.update!(ref_meanflow, ref_sim.flow)
        next_save = sim_time(sim) + save_interval
        save_velocity_frame!(gif_frames, sim, sim_time(sim))
    end

    step += 1
end

print_metrics(res; pred_label="(flexible OOD)", avg_steps_per_pred=isempty(n_integrs) ? nothing : mean(n_integrs))

plt_combined = plot_accel_combined(res, t_train_plot, t_test_plot, t_accel_end)
rst_comp_plot = plot_rst_comparison(sim_meanflow, ref_meanflow)
plt_meanflow = plot_meanflow_comparison(sim_meanflow, ref_meanflow)
save_accel_plots(savedir, plt_combined, rst_comp_plot, plt_meanflow)

if !isempty(gif_frames)
    create_velocity_gif(gif_frames, savedir)
end

println("AE checkpoint: $(AE_path)")
println("NODE checkpoint: $(node_path)")
println("Saved outputs to: $(savedir)")

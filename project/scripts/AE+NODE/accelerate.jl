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

# 200 epochs no physics in loss func
# node_path = "data/saved_models/NODE/16/RE2500/E200_MS_Adam_250/node_params.jld2"
# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E200_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2"
# aenode = AENODE(AE_path, node_path)

# 1000 no physics in loss func
# node_path = "data/saved_models/NODE/16/RE2500/E1000_MS_Adam_250/node_params.jld2"
# AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2"
# aenode = AENODE(AE_path, node_path)

# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)

@show typeof(simdata.force), typeof(simdata.time)

# random_int = findfirst(t -> t > aenode.ae_args.t_training, simdata.time)
random_int = 100
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)

train_idx, val_idx, test_idx = Thesis.get_idxs(simdata, aenode.ae_args)
t_train = simdata.time[train_idx]
t_test = simdata.time[test_idx]

t_end = 50
n_pred = 10
n_switch = 200
with_pred = true

forces_wat  = Vector{Vector{Float32}}()
time_wat   = Float32[]

forces_preds = Vector{Vector{Float32}}()
time_pred = Float32[]

step = 0
@timeit to "run simulation" begin
    while sim_time(sim) < t_end
        # sim_info(sim)
        if (step % n_switch == 0)
            raw_force_wat= WaterLily.pressure_force(sim)
            scaled_force_wat= Float32.(raw_force_wat./(0.5sim.L*sim.U^2)) # scale the forces!
            push!(forces_wat, scaled_force_wat)
            push!(time_wat, Float32(round(sim_time(sim),digits=4)))

            if with_pred
                @timeit to "predict $n_pred timesteps" begin
                    sim = predict_n(aenode, sim, n_pred; Δt=mean(sim.flow.Δt[n_pred:end-1]))
                    @info "  inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3))"
                end

                raw_force_pred = WaterLily.pressure_force(sim)
                scaled_force_pred = Float32.(raw_force_pred./(0.5sim.L*sim.U^2)) # scale the forces!
                push!(forces_preds, scaled_force_pred)
                push!(time_pred, Float32(round(sim_time(sim),digits=4)))
            end
        else
            @timeit to "sim_step" sim_step!(sim;)
            @info "WaterLily step: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3))"
        end
        # raw_force = WaterLily.pressure_force(sim)
        # scaled_force = Float32.(raw_force./(0.5sim.L*sim.U^2)) # scale the forces!
        # push!(forces, scaled_force)
        # push!(time, Float32(round(sim_time(sim),digits=4)))
        step +=1
    end
end

show(to)

# with_pred = train_force_plot(forces, time)
plt = Thesis.train_force_plot(simdata.force, simdata.time; train_idx=train_idx, test_idx=test_idx, show_zeros=false)   

# waterlily_drag, waterlily_lift = first.(forces_wat), last.(forces_wat)
# plt = plot(time_wat, waterlily_drag, label="drag", color=:red, linewidth=1.5, linestyle=:solid)
# plot!(plt, time_wat, waterlily_lift, label="life", color=:blue, linewidth=1.5, linestyle=:solid)

pred_drag, pred_lift = first.(forces_preds), last.(forces_preds)
plot!(plt, time_pred, pred_drag, label="pred drag", color=:red, linewidth=3, linestyle=:dash)
plot!(plt, time_pred, pred_lift, label="pred life", color=:blue, linewidth=3, linestyle=:dash)

Thesis.region_spans!(plt, t_train, t_test)

# plot!(plt, time, [pred_drag, pred_lift],
#         labels=["pred drag" "pred lift"],
#         colors=[:red, :blue],
#         linewidth=1.7)
display(plt)

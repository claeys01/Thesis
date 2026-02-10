using Revise
using Random
using Plots
Random.seed!(42)
includet("AENODE.jl")


using TimerOutputs
const to = TimerOutput()

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=true)

# load aenode struct with trained neural ai models
node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)


# create simulation object with flow field from training data#   load AE data
simdata = load_simdata(aenode.ae_args.full_data_path)

@show typeof(simdata.force), typeof(simdata.time)

# random_int = findfirst(t -> t > aenode.ae_args.t_training, simdata.time)
random_int = 100
u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

sim.flow.u .= u
append!(sim.flow.Δt, simdata.Δt[1:random_int])
sim_step!(sim)

train_idx, val_idx, test_idx = get_idxs(simdata, aenode.ae_args)

t_end = 50
n_pred = 5
n_switch = 10
with_pred = true


forces  = Vector{Vector{Float32}}()
time   = Float32[]

wl_forces = Vector{Vector{Float32}}()
wl_time = Float32[]

step = 0
@timeit to "run simulation" begin
    while sim_time(sim) < t_end
        # sim_info(sim)
        if (step % n_switch == 0) && with_pred
            @timeit to "predict $n_pred timesteps" begin
                sim = predict_n(aenode, sim, n_pred; Δt=mean(sim.flow.Δt[n_pred:end-1]))
                @info "  inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3))"
            end
            raw_force = WaterLily.pressure_force(sim)
            scaled_force = Float32.(raw_force./(0.5sim.L*sim.U^2)) # scale the forces!
            push!(forces, scaled_force)
            push!(time, Float32(round(sim_time(sim),digits=4)))
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
plt = train_force_plot(simdata.force, simdata.time; train_idx=train_idx, test_idx=test_idx, show_zeros=false)   

pred_drag, pred_lift = first.(forces), last.(forces)
plot!(plt, time, pred_drag, label="pred drag", color=:red, linewidth=1.5, linestyle=:dash)
plot!(plt, time, pred_lift, label="pred life", color=:blue, linewidth=1.5, linestyle=:dash)

# plot!(plt, time, [pred_drag, pred_lift],
#         labels=["pred drag" "pred lift"],
#         colors=[:red, :blue],
#         linewidth=1.7)
display(plt)

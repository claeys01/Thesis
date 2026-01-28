using Revise

includet("AENODE.jl")

using TimerOutputs
const to = TimerOutput()

sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=true)


node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"

aenode = AENODE(AE_path, node_path)


nsteps = 50
n_pred = 5

@timeit to "run simulation" begin
    for step in 1:nsteps
        if step % n_pred == 0
            @timeit to "predict $n_pred timesteps" begin
                Δt_pred = mean(sim.flow.Δt[end-(n_pred-1):end])
                û = predict_n(aenode, sim, n_pred; Δt=Δt_pred, return_traj=true)
                sim.flow.u .= û[:, :, :, end]
                append!(sim.flow.Δt, [Δt_pred for i in 1:n_pred])
                @info "     inserted prediction for $n_pred steps: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3))"

            end
        else
            @timeit to "sim_step" sim_step!(sim; verbose=true)
            @info "WaterLily step: tU/L=$(round(sim_time(sim),digits=4)), Δt=$(round(sim.flow.Δt[end],digits=3))"

        end
    end
end

show(to)


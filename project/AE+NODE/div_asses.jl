using Revise
using Random
using Plots

Random.seed!(42)
includet("AENODE.jl")


# load aenode struct with trained neural ai models
node_path = "data/saved_models/NODE/16/RE2500/multiple_shoot_adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
aenode = AENODE(AE_path, node_path)

simdata = load_simdata(aenode.ae_args.full_data_path)

u = simdata.u[:, :, :, 1]
μ₀ = simdata.μ₀[:, :, :, 1]

time = simdata.time
t = time[1]

div_u = div_field(u; buff=2)

pred_div_arr = []
running_pred_div_arr = []
rollout = []
div_arr = []
pred_time = []
for n in 1:10:4000
    û = predict_n(aenode, simdata.u[:, :, :, n], μ₀, 1, t; return_traj=false, impose_biot=false)
    div_û = div_field(û; buff=2)
    mean_div_û = round(mean(div_û); digits=8)
    append!(pred_div_arr, mean_div_û)
    append!(div_arr, mean(div_field(simdata.u[:, :, :, n] .* μ₀; buff=2)))
    append!(pred_time, time[n])
    append!(rollout, n)
    append!(running_pred_div_arr, mean(pred_div_arr))
    @info "nₜ = $(n): ‖∇·û‖ = $(mean_div_û), tₚ=$(time[n]), running avg: $(mean(pred_div_arr))"
end

@show mean(pred_div_arr)
plt = plot(pred_time, pred_div_arr; label="‖∇·û‖", color=:blue, grid=true)
plot!(plt, pred_time, running_pred_div_arr; label = "‖∇·û‖ (running avg)", color=:blue, linestyle=:dash)
plot!(plt, pred_time, div_arr; label="‖∇·u‖", color=:red)
# plt = plot(rollout, pred_div_arr; label="‖∇·û‖", color=:blue, grid=true)
# plot!(plt, rollout, running_pred_div_arr; label = "‖∇·û‖ (running avg)", color=:blue, linestyle=:dash)
# plot!(plt, rollout, div_arr; label="‖∇·u‖", color=:red)


plot!(plt; gridalpha=0.3, gridcolor=:gray, xgrid=true, ygrid=true)

display(plt)
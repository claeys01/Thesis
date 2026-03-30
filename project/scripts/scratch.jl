using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Plots
using TimerOutputs

# sim = circle_shedding_biot(;mem=Array, Re=2500, n=2^8, m=2^8, perturb=false, Δt=0.0)
reset_timer!(to::TimerOutput)

# 1000 with physics in loss func
node_path = "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2"
AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"
# aenode = AENODE(AE_path, node_path)

# simdata = load_simdata(aenode.ae_args.full_data_path)
# simdata = load_simdata("data/datasets/RE2500/2e8/U_128_transfer.jld2")
# @show simdata.time[1], simdata.time[end]


if is_hpc()
    root_path = "scratch/mfbclaeys"
    println("is hpc")
else
    root_path = ""
end
# node_path = joinpath(root_path, "data/saved_models/NODE/16/RE2500/E1000_curldiv_MS_Adam_250/node_params.jld2")
AE_path = joinpath(root_path, "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2")
tl_path = joinpath(root_path, "data/datasets/RE2500/2e8/U_128_transfer.jld2")

# retraindata = simdata = load_simdata(tl_path)
# @show simdata.time[end] simdata.time[1]

# idx_train = argmin(abs.(simdata.time .- aenode.ae_args.t_training))
# idx_30 = argmin(abs.(simdata.time .- 30))
# transfer_span = collect(idx_train:idx_30)


# @show size(simdata.period_ranges)
# @show simdata.force[transfer_span]
# simdata = SimData(
#     simdata.time[transfer_span],
#     simdata.Δt[transfer_span],
#     simdata.u[:, :, :, transfer_span],
#     simdata.p[:, :, transfer_span],
#     simdata.f[:, :, :, transfer_span],
#     simdata.μ₀[:, :, :, transfer_span],
#     simdata.force[transfer_span],
#     simdata.ε[transfer_span],
#     simdata.period_ranges,
#     simdata.reordered_ranges,
#     simdata.single_period_idx,
# )

# @save "data/datasets/RE2500/2e8/U_128_transfer.jld2" simdata

# random_int = 1
# u, μ₀, t₀ = simdata.u[:, :, :, random_int], simdata.μ₀[:, :, :, random_int], simdata.time[random_int]

# sim.flow.u .= u
# # append!(sim.flow.Δt, simdata.Δt[1:random_int])
# # sim_step!(sim)
# # sim_step!(sim)
# # f = deepcopy(sim.flow.f)
# # @show size(f)
# # H, B, C = size(f)

# # for c in 1:C
# #     for j in 1:B
# #         for i in 1:H
# #             elem = f[i, j, c]
# #             # @show typeof(elem), elem

# #             if isnan(elem)
# #                 @show "jemoder"
# #                 f[i, j, c] = -100
# #             end
# #         end
# #     end
# # end

# # # f_clean = replace(f, Int64(NaN) => 0.0)
# # display(WaterLily.flood(f[:, :, 1]))

# sim_meanflow = MeanFlow(sim.flow; uu_stats=true)
# t_end = 50
# step = 1

# p_list = Vector{Array{Float32,2}}()
# f_list = Vector{Array{Float32,3}}()
# for i in eachindex(simdata.time)
#     push!(sim.flow.Δt, simdata.Δt[i])
#     sim_step!(sim)
#     # display(WaterLily.flood(sim.flow.f[:, :, 1]))


#     println("-"^10, step, "-"^10)
#     current_time, current_Δt = sim_time(sim), sim.flow.Δt[end]
#     saved_time, saved_Δt   = simdata.time[i], simdata.Δt[i]
    
#     println("sim:   tU/L=$(current_time), Δt=$(current_Δt)")
#     println("saved: tU/L=$(saved_time), Δt=$(saved_Δt)")

#     # @show mean(sim.flow.p)
#     # display(WaterLily.flood(sim.flow.p))
#     push!(p_list, copy(sim.flow.p))
#     push!(f_list, copy(sim.flow.f))
#     # @assert current_Δt == saved_Δt
#     # @assert current_time == saved_time


#     step += 1
# end
# p = cat(p_list...; dims = 3) 
# f = cat(p_list...; dims = 4) 
# H, W, T = size(p)

# simdata = SimData(
#     simdata.time,
#     simdata.Δt,
#     simdata.u,
#     p,
#     f,
#     simdata.μ₀,
#     simdata.force,
#     simdata.ε,
#     simdata.period_ranges,
#     simdata.reordered_ranges,
#     simdata.single_period_idx,
# )
# # @save "data/datasets/RE2500/2e8/U_128_full.jld2" simdata
# @show T, size(simdata.time)
# @assert T == size(simdata.time)[1]
# @show simdata.time[1], simdata.time[end]
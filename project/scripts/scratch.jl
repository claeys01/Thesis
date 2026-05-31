using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots
using Printf

simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")

u_0_path = save_u0("data/initial_fields/RE2500/2e8/u_0.jld2", simdata.u[:, :, :, 1])
@show size(simdata.u[:, :, :, 1])

# root_path = ""
# if is_hpc()
#     root_path = "/scratch/mfbclaeys"
#     # Log job info
#     @info "Starting HPC AE+NODE retrain pipeline"
#     @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
#     @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
#     @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
#     @info "  Hostname: $(gethostname())"
#     @info "  Julia threads: $(Threads.nthreads())"
# end

# params = params = InlineParams(
#         t_run = 50, 
#         t_train = 16.603,
#         t_accel_end = 50,
#         ae_epochs = 500,
#         ae_retrain_epochs = 100,
#         node_iters = 250,
#         node_retrain_iters = 100,
#         n_switch = 150,
#         max_retrain_flags = 3,
#         save_interval = 0.25, # needs to be fixed still, 
#     )


# savedir = joinpath(root_path, "data", "inline_runs", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
# mkpath(savedir)
# # simdata_path = joinpath(savedir, "U_inline.jld2")

# # u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
# sim = circle_shedding_biot(; Re=250, mem=Array, perturb=true)
# sim_step!(sim, 50; verbose=true)
# u₀ = sim.flow.u
# @save "data/initial_fields/RE250/2e8/u_0.jld2" u₀
# datasets
# hs = HybridState(sim, nothing, params, savedir, nothing, nothing)

# simdata = run_warmup!(hs, params.t_run)

# # simdata = load_simdata("data/datasets/RE2500/2e8/U_128_full.jld2")
# # display(Thesis.train_force_plot(simdata))
# # default(fontfamily="Computer Modern", titlefontsize=14,
# #         guidefontsize=12, tickfontsize=8, legendfontsize=9)
# plt = plot(framestyle=:box, size=(600, 300), dpi=500,
#         xlabel="\$t^*\$", ylabel="Force coefficient",
#         xlims=(0, 50), ylims=(-3, 2))
# plot!(simdata.time, first.(simdata.force), label=L"C_{d}", color=:red, lw=1)
# plot!(simdata.time, last.(simdata.force), label=L"C_{L}", color=:blue, lw=1)
# display(plt)
# savefig(plt, "figs/256_forces.pdf")

# @show size(simdata.u)
# println(dump(simdata))

# n = 2^8
# sim = circle_shedding_biot(;n=n, m=n, Re=2500)

# sim_step!(sim, )

# u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
# sim.flow.u .= u₀


# next_delta = 0.5
# next_plot = copy(sim_time(sim)) + next_delta
# counter = 1
# t_end=10
# while sim_time(sim) < t_end
#     sim_step!(sim)
#     if next_plot < sim_time(sim)
#         sim_info(sim)
#         @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#         @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])

#         plt = flood(sim.flow.σ,shift=(-2,-1.5),clims=(-8,8), axis=([], false),  
#         background=:gray,
#         cfill=:seismic,legend=false,border=:none,dpi=350, size=(800, 800))
#         bod = body_plot!(sim)
#         timestep = counter * next_delta
#         display(plt)
#         savefig(plt, "figs/biot_shedding_plots/shedding_t$timestep.png")
#         next_plot += next_delta
#         counter +=1
#     end
# end

# function run_oscillating_flow(n=2^9, stop=20)
#     sim = circle_shedding_biot(;n=n,m=n)
#     sim_step!(sim,0.1)

#     @time @gif for tᵢ in range(0.,stop;step=0.2)
#         println("tU/L=",round(tᵢ,digits=4))
#         sim_step!(sim,tᵢ)
#         @inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
#         @inside sim.flow.σ[I] = ifelse(abs(sim.flow.σ[I])<0.001,0.0,sim.flow.σ[I])
#         # It's important to have `|>Array` during GPU simulation as `flood` only accept CPU Array input
#         flood(sim.flow.σ|>Array,shift=(-2,-1.5),clims=(-8,8), axis=([], false),
#               cfill=:seismic,legend=false,border=:none,size=(800,800), dpi=350)
#         body_plot!(sim)
#     end
# end

# run_oscillating_flow()
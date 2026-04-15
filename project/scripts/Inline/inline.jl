using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots


# Small container for the main timing/training knobs of the experiment.
Base.@kwdef struct InlineParams
    t_run = 20.0            # length to run simulation for
    t_train = 16.3          # amount of time used for trainingn, rest is test data
    t_accel_end = 50        # when to stop hybrid simulation
    ae_epochs = 1
    node_iters = 250        
    n_switch = 100
    pred_Δt = 0.35
    save_interval = 2
    max_retrain_flags = 3
end

Base.@kwdef mutable struct HybridState
    sim::BiotSimulation
    ref_sim::BiotSimulation
    aenode::AENODE
    params::InlineParams
    sim_meanflow::MeanFlow
    ref_meanflow::MeanFlow
    res::AccelResults
    n_integrs::Vector{Int}
    gif_frames::Vector{Any}
    t_train_plot::Vector{Float64}
    t_test_plot::Vector{Float64}
    savedir::String
    AE_path::String
    node_path::String
    simdata_path::String = ""
    step::Int = 1
    next_save::Float64 = 2.0
    retrain_needed::Bool = false
end

function HybridState(sim, aenode, params, savedir, AE_path, node_path)
    ref_sim = deepcopy(sim)
    sim_meanflow = MeanFlow(sim.flow; uu_stats=true)
    ref_meanflow = MeanFlow(ref_sim.flow; uu_stats=true)

    return HybridState(;
        sim, ref_sim, aenode, params,
        sim_meanflow, ref_meanflow,
        res=AccelResults(),
        n_integrs=Int[],
        gif_frames=[],
        t_train_plot=Float64[],
        t_test_plot=Float64[],
        savedir, AE_path, node_path,
        next_save=Float64(params.save_interval),
    )
end

function run_warmup!(hs::HybridState, t_end; u₀=nothing, save_path=nothing)
    (; sim, ref_sim, res) = hs

    if !isnothing(u₀) && size(sim.flow.u) == size(u₀)
        sim.flow.u .= u₀
        ref_sim.flow.u .= u₀
    end

    time_vec = Float32[]
    Δt_vec = Float32[]
    u_list = Vector{Array{Float32,3}}()
    p_list = Vector{Array{Float32,2}}()
    μ₀_list = Vector{Array{Float32,3}}()
    f_list = Vector{Array{Float32,3}}()
    force_list = Vector{Vector{Float32}}()

    @info "Starting warmup simulation" t_end
    while sim_time(sim) < t_end
        wall_time = @elapsed sim_step!(sim)
        record_waterlily_step!(res, sim, wall_time)

        while sim_time(ref_sim) < sim_time(sim)
            step_reference!(res, ref_sim)
        end

        push!(u_list, copy(sim.flow.u))
        push!(p_list, copy(sim.flow.p))
        push!(μ₀_list, copy(sim.flow.μ₀))
        push!(f_list, copy(sim.flow.f))
        push!(force_list, copy(res.hybrid_forces_wat[end]))
        push!(time_vec, res.hybrid_time_wat[end])
        push!(Δt_vec, Float32(round(sim.flow.Δt[end], digits=3)))

        hs.step += 1
    end
    @info "Warmup complete" sim_time=sim_time(sim) steps=hs.step-1

    simdata = SimData(
        time=time_vec, Δt=Δt_vec,
        u=cat(u_list...; dims=4),
        p=cat(p_list...; dims=3),
        f=cat(f_list...; dims=4),
        μ₀=cat(μ₀_list...; dims=4),
        force=force_list,
        ε=Float32[],                        # just leave empty because not used
        period_ranges=UnitRange{Int}[],     # just leave empty because not used
        reordered_ranges=UnitRange{Int}[],  # just leave empty because not used
        single_period_idx=1:0,              # just leave empty because not used
    )

    if !isnothing(save_path)
        @save save_path simdata
        hs.simdata_path = save_path
        @info "Saved simdata to $save_path"
    end

    train_idx, _, test_idx = Thesis.get_idxs(simdata, hs.aenode.ae_args)
    hs.t_train_plot = simdata.time[train_idx]
    hs.t_test_plot = simdata.time[test_idx]

    hs.sim_meanflow = MeanFlow(hs.sim.flow; uu_stats=true)
    hs.ref_meanflow = MeanFlow(hs.ref_sim.flow; uu_stats=true)

    return simdata
end

function run_hybrid!(hs::HybridState)
    (; sim, ref_sim, aenode, params, sim_meanflow, ref_meanflow,
       res, n_integrs, gif_frames) = hs
    retrain_req_counter = 0

    predict_flex(aenode, deepcopy(sim); Δt=Float32(params.pred_Δt), impose_biot=true, verbose=false) # warmup predict function 

    while sim_time(sim) < params.t_accel_end
        if hs.step % params.n_switch == 0 && sim_time(sim) > aenode.ae_args.t_training
            if retrain_req_counter ≥ params.max_retrain_flags
                @warn "Latent trajectory exceeded limit too many times, AE and NODE retraining needed"
                hs.retrain_needed = true
                return hs
            end

            sim_time_before = sim_time(sim)
            predict_wall_time = @elapsed begin
                sim, n_integr, retrain_required = predict_flex(aenode, sim; Δt=Float32(params.pred_Δt), impose_biot=true, next_save=hs.next_save)
            end
            retrain_required && (retrain_req_counter += 1)
            sim_dt = sim_time(sim) - sim_time_before

            if n_integr != 0
                push!(n_integrs, n_integr)
                record_prediction!(res, sim, predict_wall_time, sim_dt, hs.step)
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

        if sim_time(sim) > hs.next_save
            WaterLily.update!(sim_meanflow, sim.flow)
            WaterLily.update!(ref_meanflow, ref_sim.flow)
            hs.next_save = sim_time(sim) + params.save_interval
            save_velocity_frame!(gif_frames, sim, sim_time(sim))
        end

        hs.step += 1
    end

    hs.retrain_needed = false
    return hs
end

function save_results(hs::HybridState)
    (; res, n_integrs, t_train_plot, t_test_plot, params, savedir,
       sim_meanflow, ref_meanflow, gif_frames, AE_path, node_path) = hs

    print_metrics(res; pred_label="(flexible OOD)",
                  avg_steps_per_pred=isempty(n_integrs) ? nothing : mean(n_integrs))

    plt_combined = plot_accel_combined(res, t_train_plot, t_test_plot, params.t_accel_end)
    display(plt_combined)
    rst_comp_plot = plot_rst_comparison(sim_meanflow, ref_meanflow)
    plt_meanflow = plot_meanflow_comparison(sim_meanflow, ref_meanflow)
    save_accel_plots(savedir, plt_combined, rst_comp_plot, plt_meanflow)

    if !isempty(gif_frames)
        create_velocity_gif(gif_frames, savedir)
    end

    println("AE checkpoint: $(AE_path)")
    println("NODE checkpoint: $(node_path)")
    println("Saved outputs to: $(savedir)")
end

params = InlineParams()

savedir = joinpath("data", "inline_runs", Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline.jld2")

u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")
sim = circle_shedding_biot(; mem=Array, perturb=false)

AE_path_tl1 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL1_E500_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
normalizer = load_normalizer(AE_path_tl1)
ae_bundle, ae_args = load_trained_AE(AE_path_tl1)

node_path = "data/saved_models/NODE/16/RE2500/TL1_E500_curldiv_MS_Adam_250/node_params.jld2"
node, node_args = load_node(node_path)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

hs = HybridState(sim, aenode, params, savedir, AE_path_tl1, node_path)
simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path)
run_hybrid!(hs)

if hs.retrain_needed
    @info "Retraining triggered at sim_time=$(sim_time(hs.sim)), step=$(hs.step)"

    AE_path_tl2 = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/TL2_E300_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p0002_wd0p0009_bs16_NY_LL1_Tl0p0/checkpoint.jld2"
    normalizer = load_normalizer(AE_path_tl2)
    ae_bundle, ae_args = load_trained_AE(AE_path_tl2)

    node_path_tl2 = "data/saved_models/NODE/16/RE2500/TL2_E300_curldiv_MS_Adam_250/node_params.jld2"
    node, node_args = load_node(node_path_tl2)

    hs.aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)
    hs.AE_path = AE_path_tl2
    hs.node_path = node_path_tl2
    hs.retrain_needed = false
    run_hybrid!(hs)
end

save_results(hs)

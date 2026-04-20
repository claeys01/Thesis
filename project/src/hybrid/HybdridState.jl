Base.@kwdef struct InlineParams
    t_run = 10
    t_train = 8
    t_accel_end = 25
    ae_epochs = 1
    node_iters = 50
    n_switch = 150
    pred_Δt = 0.35
    save_interval = 1
    max_retrain_flags = 3
end

Base.@kwdef mutable struct HybridState
    sim::BiotSimulation
    ref_sim::BiotSimulation
    aenode::Union{AENODE, Nothing}
    params::InlineParams
    sim_meanflow::MeanFlow
    ref_meanflow::MeanFlow
    res::AccelResults
    n_integrs::Vector{Int}
    gif_frames::Vector{Any}
    savedir::String
    AE_path::Union{String, Nothing}
    node_path::Union{String, Nothing}
    simdata_path::String = ""
    step::Int = 1
    next_save::Float32 = 0
    retrain_needed::Bool = false
    mode_log::Vector{@NamedTuple{t_start::Float32, t_end::Float32, mode::String}} = @NamedTuple{t_start::Float32, t_end::Float32, mode::String}[]
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
        savedir, AE_path, node_path,
        next_save=Float64(params.save_interval),
    )
end

function run_warmup!(hs::HybridState, t_end; simdata::Union{SimData,Nothing}=nothing, u₀=nothing, save_path=nothing, verbose=true)
    (; sim, ref_sim, res, mode_log) = hs

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

    t_sim_start = sim_time(sim)

    verbose && @info "Starting warmup simulation" t_end continuing = !isnothing(simdata)
    while sim_time(sim) < t_end
        wall_time = @elapsed sim_step!(sim)
        record_waterlily_step!(res, sim, wall_time)

        while sim_time(ref_sim) < sim_time(sim)
            step_reference!(res, ref_sim)
        end

        if sim_time(sim) > hs.next_save
            WaterLily.update!(hs.sim_meanflow, sim.flow)
            WaterLily.update!(hs.ref_meanflow, ref_sim.flow)
            hs.next_save = sim_time(sim) + hs.params.save_interval
            save_velocity_frame!(hs.gif_frames, sim, sim_time(sim))
            verbose && @info "Updating MeanFlow statistics at: $(sim_time(sim))"
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
    verbose && @info "Warmup complete" sim_time = sim_time(sim) steps = hs.step - 1

    push!(mode_log, (t_start=t_sim_start, t_end=time_vec[end], mode="Training"))

    new_u = cat(u_list...; dims=4)
    new_p = cat(p_list...; dims=3)
    new_f = cat(f_list...; dims=4)
    new_μ₀ = cat(μ₀_list...; dims=4)

    if isnothing(simdata)
        simdata = SimData(
            time=time_vec, Δt=Δt_vec,
            u=new_u, p=new_p, f=new_f, μ₀=new_μ₀,
            force=force_list,
            ε=Float32[],
            period_ranges=UnitRange{Int}[],
            reordered_ranges=UnitRange{Int}[],
            single_period_idx=1:0,
        )
    else
        new_u = clip_time_series(new_u)
        new_μ₀ = clip_time_series(new_μ₀)
        append!(simdata.time, time_vec)
        append!(simdata.Δt, Δt_vec)
        simdata.u = cat(simdata.u, new_u; dims=4)
        simdata.p = cat(simdata.p, new_p; dims=3)
        simdata.f = cat(simdata.f, new_f; dims=4)
        simdata.μ₀ = cat(simdata.μ₀, new_μ₀; dims=4)
        append!(simdata.force, force_list)
    end

    if !isnothing(save_path)
        @save save_path simdata
        hs.simdata_path = save_path
        @info "Saved simdata to $save_path"
    end

    # train_idx, _, test_idx = Thesis.get_idxs(simdata, hs.aenode.ae_args)
    # hs.t_train_plot = simdata.time[train_idx]
    # hs.t_test_plot = simdata.time[test_idx]

    # hs.sim_meanflow = MeanFlow(hs.sim.flow; uu_stats=true)
    # hs.ref_meanflow = MeanFlow(hs.ref_sim.flow; uu_stats=true)

    return simdata
end

function run_hybrid!(hs::HybridState; verbose=true)
    (; sim, ref_sim, aenode, params, sim_meanflow, ref_meanflow,
        res, n_integrs, gif_frames, mode_log) = hs
    retrain_req_counter = 0

    predict_flex(aenode, deepcopy(sim); Δt=Float32(params.pred_Δt), impose_biot=true, verbose=false) # warmup predict function 
    t_hybrid_start = sim_time(sim)

    while sim_time(sim) < params.t_accel_end
        # if hs.step % params.n_switch == 0 && sim_time(sim) > aenode.ae_args.t_training
        if hs.step % params.n_switch == 0
            if retrain_req_counter ≥ params.max_retrain_flags
                @info "Latent trajectory exceeded limit too many times, retraining AE and NODE at $(sim_time(sim))"
                hs.retrain_needed = true
                push!(mode_log, (t_start=t_hybrid_start, t_end=sim_time(sim), mode="Hybrid"))
                return hs
            end

            sim_time_before = sim_time(sim)
            predict_wall_time = @elapsed begin
                sim, n_integr, retrain_required = predict_flex(aenode, sim; Δt=Float32(params.pred_Δt), impose_biot=true, next_save=hs.next_save)
            end
            if retrain_required
                retrain_req_counter += 1
                push!(mode_log, (t_start=sim_time(sim), t_end=sim_time(sim), mode="Retrain flag"))
            end
            sim_dt = sim_time(sim) - sim_time_before

            if n_integr != 0
                push!(n_integrs, n_integr)
                record_prediction!(res, sim, predict_wall_time, sim_dt, hs.step)
                println(" Inserted prediction for $n_integr steps: tU/L=$(round(sim_time(sim), digits=4)), wall time: $(round(predict_wall_time*1000, digits=4)) ms, force: $(res.hybrid_forces_preds[end])")
            else
                @info "nothing inserted: $(sim_time(sim))"
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
            @info "Updating MeanFlow statistics at: $(sim_time(sim))"
        end

        hs.step += 1
    end

    push!(mode_log, (t_start=t_hybrid_start, t_end=sim_time(sim), mode="Hybrid"))

    hs.retrain_needed = false
    return hs
end

function save_results(hs::HybridState)
    (; res, n_integrs, params, savedir,
        sim_meanflow, ref_meanflow, gif_frames, AE_path, node_path, mode_log) = hs

    print_metrics(res; pred_label="(flexible OOD)",
        avg_steps_per_pred=isempty(n_integrs) ? nothing : mean(n_integrs))

    plt_combined = plot_accel_combined(res, params.t_accel_end; mode_log=mode_log)
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
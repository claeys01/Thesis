Base.@kwdef struct InlineParams
    t_run = 20
    t_train = 16.603
    t_accel_end = 50
    t_update = 1
    ae_epochs = 400
    ae_retrain_epochs = 100
    ae_batch_size = 8
    node_iters = 250
    continuity_term = 250
    node_retrain_iters = 150
    continuity_term_retrain = 500
    downsample=500
    group_size=20
    n_switch = 150
    pred_Δt = 0.35
    save_interval = 0.05
    sample_interval = 0.0
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
    next_sample::Float32 = 0
    retrain_needed::Bool = false
    mode_log::Vector{@NamedTuple{t_start::Float32, t_end::Float32, mode::String}} = @NamedTuple{t_start::Float32, t_end::Float32, mode::String}[]
    save_fields::Bool = true
    n_field_saved::Int = 0
    hybrid_field_file::Any = nothing
    ref_field_file::Any = nothing
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
        next_sample=Float64(params.sample_interval),
    )
end

function run_warmup!(hs::HybridState, t_end; simdata::Union{SimData,Nothing}=nothing, u₀=nothing, save_path=nothing, 
    verbose=true, run_ref=true, save_gif=false)
    (; sim, ref_sim, res, mode_log) = hs
    
    @show size(sim.flow.u), size(u₀)
    if !isnothing(u₀) && size(sim.flow.u) == size(u₀)
        @info "Initial condition set"
        sim.flow.u .= u₀
        ref_sim.flow.u .= u₀
    end

    time_vec = Float32[]
    Δt_vec = Float32[]
    u_list = Vector{Array{Float32,3}}()
    p_list = Vector{Array{Float32,2}}()
    μ₀_list = Vector{Array{Float32,3}}()
    force_list = Vector{Vector{Float32}}()

    t_sim_start = sim_time(sim)

    verbose && @info "Starting warmup simulation" t_end continuing = !isnothing(simdata)
    while sim_time(sim) < t_end && sim_time(sim) < hs.params.t_accel_end
        wall_time = @elapsed begin
            sim_step!(sim)
            # sync_device!()
        end
        # record_waterlily_step!(res, sim, wall_time)

        while sim_time(ref_sim) < sim_time(sim) && run_ref
            step_reference!(res, ref_sim)
        end

        if sim_time(sim) > hs.next_save
            WaterLily.update!(hs.sim_meanflow, sim.flow)
            run_ref && WaterLily.update!(hs.ref_meanflow, ref_sim.flow)
            hs.next_save = sim_time(sim) + hs.params.save_interval
            save_gif && save_velocity_frame!(hs.gif_frames, sim, sim_time(sim))
            verbose && @info "  Updating MeanFlow statistics at: $(sim_time(sim))"
        end

        if sim_time(sim) ≥ hs.next_sample
            record_waterlily_step!(res, sim, wall_time)

            push!(u_list, copy(sim.flow.u))
            push!(p_list, copy(sim.flow.p))
            push!(μ₀_list, copy(sim.flow.μ₀))
            push!(force_list, copy(res.hybrid_forces_wat[end]))
            push!(time_vec, res.hybrid_time_wat[end])
            push!(Δt_vec, Float32(round(sim.flow.Δt[end], digits=3)))
            hs.next_sample = sim_time(sim) + hs.params.sample_interval
            # verbose && @info "Updating simdata statistics at: $(sim_time(sim))"
        end

        run_ref && save_field_step!(hs, sim, ref_sim)

        hs.step += 1

    end
    verbose && @info "Warmup complete" sim_time = sim_time(sim) steps = hs.step - 1 samples = length(time_vec)

    if isempty(time_vec)
        @warn "run_warmup! collected no samples — sample_interval $(hs.params.sample_interval) exceeds the warmup window"
        push!(mode_log, (t_start=t_sim_start, t_end=Float32(sim_time(sim)), mode="Training"))
        return simdata
    end

    push!(mode_log, (t_start=t_sim_start, t_end=time_vec[end], mode="Training"))

    new_u = cat(u_list...; dims=4)
    new_p = cat(p_list...; dims=3)
    new_μ₀ = cat(μ₀_list...; dims=4)

    if isnothing(simdata)
        simdata = SimData(
            time=time_vec, Δt=Δt_vec,
            u=new_u, p=new_p, μ₀=new_μ₀,
            force=force_list,
            chunk_ranges=[1:length(time_vec)],
        )
    else
        # new_u = clip_time_series(new_u)
        # new_μ₀ = clip_time_series(new_μ₀)
        prev_end = length(simdata.time)
        append!(simdata.time, time_vec)
        append!(simdata.Δt, Δt_vec)
        try
            simdata.u = cat(simdata.u, new_u; dims=4)
            simdata.μ₀ = cat(simdata.μ₀, new_μ₀; dims=4)
        catch e
            new_u = clip_time_series(new_u)
            new_μ₀ = clip_time_series(new_μ₀)
            simdata.u = cat(simdata.u, new_u; dims=4)
            simdata.μ₀ = cat(simdata.μ₀, new_μ₀; dims=4)
        end
        simdata.p = cat(simdata.p, new_p; dims=3)
        append!(simdata.force, force_list)
        if !isdefined(simdata, :chunk_ranges) || isempty(simdata.chunk_ranges)
            simdata.chunk_ranges = [1:prev_end]
        end
        push!(simdata.chunk_ranges, (prev_end+1):length(simdata.time))
    end

    if !isnothing(save_path)
        @save save_path simdata
        hs.simdata_path = save_path
        @info "Saved simdata to $save_path"
    end
    u_list = nothing
    p_list = nothing
    μ₀_list = nothing
    force_list = nothing
    return simdata
end

function update_meanflow_snapshot!(meanflow::MeanFlow, u, p, t)
    T = eltype(meanflow.P)
    t_snapshot = T(t)
    dt = t_snapshot - meanflow.t[end]
    dt <= zero(T) && return meanflow

    ε = dt / (dt + WaterLily.time(meanflow) + eps(T))
    length(meanflow.t) == 1 && (ε = one(T))

    @loop meanflow.P[I] = ε * p[I] + (one(T) - ε) * meanflow.P[I] over I in CartesianIndices(meanflow.P)
    @loop meanflow.U[Ii] = ε * u[Ii] + (one(T) - ε) * meanflow.U[Ii] over Ii in CartesianIndices(meanflow.U)

    if meanflow.uu_stats
        for i in 1:ndims(meanflow.P), j in 1:ndims(meanflow.P)
            @loop meanflow.UU[I,i,j] = ε * (u[I,i] .* u[I,j]) + (one(T) - ε) * meanflow.UU[I,i,j] over I in CartesianIndices(meanflow.P)
        end
    end

    push!(meanflow.t, t_snapshot)
    return meanflow
end

function update_predicted_meanflow!(meanflow::MeanFlow, sim::BiotSimulation, û_meanflow, t_meanflow)
    if isnothing(û_meanflow) || isnothing(t_meanflow) || isempty(t_meanflow)
        return Vector{Vector{Float32}}(), Float32[]
    end

    scratch_sim = deepcopy(sim)
    snapshots = ndims(û_meanflow) == 3 ? reshape(û_meanflow, size(û_meanflow)..., 1) : û_meanflow
    pred_forces = Vector{Vector{Float32}}()
    pred_times = Float32[]

    for i in eachindex(t_meanflow)
        insert_prediction!(scratch_sim, snapshots[:, :, :, i])
        impose_biot_bc!(scratch_sim)
        update_meanflow_snapshot!(meanflow, scratch_sim.flow.u, scratch_sim.flow.p, t_meanflow[i])
        push!(pred_forces, get_forces(scratch_sim))
        push!(pred_times, Float32(t_meanflow[i]))
    end

    return pred_forces, pred_times
end

# function run_hybrid!(hs::HybridState; simdata::Union{SimData,Nothing}=nothing, save_path=nothing, verbose=true)
function run_hybrid!(hs::HybridState; verbose=true)
    (; sim, ref_sim, aenode, params, sim_meanflow, ref_meanflow,
        res, n_integrs, gif_frames, mode_log) = hs
    retrain_req_counter = 0

    predict_flex(aenode, deepcopy(sim); Δt=Float32(params.pred_Δt), impose_biot=true, verbose=false) # warmup predict function 
    t_hybrid_start = sim_time(sim)

    while sim_time(sim) < params.t_accel_end
        # if hs.step % params.n_switch == 0 && sim_time(sim) > aenode.ae_args.t_training
        if hs.step % params.n_switch == 0
            sim_time_before = sim_time(sim)
            predict_wall_time = @elapsed begin
                sim, n_integr, retrain_required, û_meanflow, t_meanflow = predict_flex(
                    aenode,
                    sim;
                    Δt=Float32(params.pred_Δt),
                    impose_biot=true,
                    next_save=hs.next_save,
                    save_interval=params.save_interval,
                )
                sync_device!()
            end
            if retrain_required
                retrain_req_counter += 1
                push!(mode_log, (t_start=sim_time(sim), t_end=sim_time(sim), mode="Retrain flag"))
            end
            sim_dt = sim_time(sim) - sim_time_before

            if n_integr != 0
                # display(Thesis.curl_plot(sim.flow.u)) 
                # display(Thesis.curl_plot(û_meanflow[:, :, :, end]))
                pred_forces, pred_times = update_predicted_meanflow!(sim_meanflow, sim, û_meanflow, t_meanflow)
                push!(n_integrs, n_integr)
                record_prediction!(
                    res,
                    sim,
                    predict_wall_time,
                    sim_dt,
                    hs.step;
                    pred_forces=pred_forces,
                    pred_times=pred_times,
                )
                println(" Inserted prediction for $n_integr steps: tU/L=$(round(sim_time(sim), digits=4)), wall time: $(round(predict_wall_time*1000, digits=4)) ms, force: $(res.hybrid_forces_preds[end])")
            else
                @info "nothing inserted: $(sim_time(sim))"
                wall_time = @elapsed begin
                    sim_step!(sim)
                    sync_device!()
                end
                # the prediction attempt cost real wall time even though nothing
                # was inserted — attribute it to the CFD step that did advance
                record_waterlily_step!(res, sim, predict_wall_time + wall_time)
            end
        else
            wall_time = @elapsed begin
                sim_step!(sim)
                sync_device!()
            end
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

        save_field_step!(hs, sim, ref_sim)

        hs.step += 1

        if retrain_req_counter ≥ params.max_retrain_flags
            @info "Latent trajectory exceeded limit too many times, retraining AE and NODE at $(sim_time(sim))"
            hs.retrain_needed = true
            push!(mode_log, (t_start=t_hybrid_start, t_end=sim_time(sim), mode="Hybrid"))
            return hs
        end
    end

    push!(mode_log, (t_start=t_hybrid_start, t_end=sim_time(sim), mode="Hybrid"))

    hs.retrain_needed = false
    return hs
end

function save_field_step!(hs::HybridState, sim, ref_sim)
    hs.save_fields || return
    if hs.hybrid_field_file === nothing
        hs.hybrid_field_file = jldopen(joinpath(hs.savedir, "U_hybrid_inline.jld2"), "w")
        hs.ref_field_file    = jldopen(joinpath(hs.savedir, "U_ref_inline.jld2"), "w")
    end
    i = (hs.n_field_saved += 1)
    hs.hybrid_field_file["u/$i"] = Array(sim.flow.u)
    hs.hybrid_field_file["t/$i"] = Float32(sim_time(sim))
    hs.ref_field_file["u/$i"]    = Array(ref_sim.flow.u)
    hs.ref_field_file["t/$i"]    = Float32(sim_time(ref_sim))
end

function close_field_files!(hs::HybridState)
    if hs.hybrid_field_file !== nothing
        hs.hybrid_field_file["n"] = hs.n_field_saved
        hs.ref_field_file["n"]    = hs.n_field_saved
        close(hs.hybrid_field_file); close(hs.ref_field_file)
        hs.hybrid_field_file = nothing; hs.ref_field_file = nothing
        @info "Saved $(hs.n_field_saved) hybrid/ref field snapshots to $(hs.savedir)"
    end
end

function save_results(hs::HybridState)
    close_field_files!(hs)
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
        # create_velocity_gif(gif_frames, savedir)
    end

    accel_path = joinpath(savedir, "accel_results.jld2")
    hs_path = joinpath(savedir, "hybrid_state.jld2")
    @save accel_path res
    @save hs_path res sim_meanflow ref_meanflow params mode_log n_integrs AE_path node_path savedir
    println("AccelResults saved to: $(accel_path)")
    println("HybridState saved to: $(hs_path)")

    println("AE checkpoint: $(AE_path)")
    println("NODE checkpoint: $(node_path)")
    println("Saved outputs to: $(savedir)")
end
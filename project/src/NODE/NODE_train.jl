make_optimiser(opt, η) = hasmethod(opt, Tuple{Float64}) ? opt(η) :
                         hasmethod(opt, Tuple{}) ? opt() :
                         error("Unsupported optimiser constructor: $(opt)")

function train_NODE(args::NodeArgs;
    ae_bundle=nothing,
    normalizer=nothing, ae_args=nothing, kws...)

    device = args.use_gpu ? get_device() : cpu_device()

    multi = false
    zs = nothing; ts = nothing; tspans = nothing; z0s = nothing

    if isnothing(ae_bundle)
        # Original path: load pre-saved latent data from disk
        z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
    else
        # check simdata to see if it is split up in chunks
        simdata_temp = !isnothing(ae_args.simdata_ram) ? ae_args.simdata_ram : load_simdata(ae_args.full_data_path)
        n_chunks = isempty(simdata_temp.chunk_ranges) ? 1 : length(simdata_temp.chunk_ranges)
        
        if isnothing(ae_args.simdata_ram)
            simdata_temp = nothing; GC.gc()
        end

        if n_chunks > 1
            multi = true
            @info "Multi-chunk simdata detected, using disjoint-trajectory NODE training" n_chunks
            zs, ts, tspans, z0s = get_latent_chunks(ae_bundle, normalizer, ae_args;
                downsample=ae_args.train_downsample, device=device,
                min_chunk_size=max(args.group_size + 1, 21))
            multi = length(zs) > 1  # in case some chunks were skipped
        end

        if !multi
            @info "Encoding latent vectors from AE in memory (no disk I/O)"
            z, t, tspan, z0 = get_latent_vectors(ae_bundle, normalizer, ae_args; device=device, downsample=ae_args.train_downsample)
            if args.downsample > 0 && args.downsample < size(z, 2)
                idx = downsample_equal(collect(1:size(z, 2)), args.downsample)
                z = z[:, idx]
                t = t[idx]
                tspan = (t[1], t[end])
                z0 = z[:, 1]
            end
        end
    end

    if multi
        # Metadata: union span and concatenated time vector. predict_flex always
        # passes explicit `t`, so `node.tspan`/`node.t` are inference-irrelevant.
        tspan = (minimum(first.(tspans)), maximum(last.(tspans)))
        t = vcat(ts...)
        z = hcat(zs...)
        z0 = z0s[1]
    end

    # z_test, t_test, tspan_test, z0_test = get_NODE_data(args.test_latent_path; downsample=args.test_downsample)

    if args.retrain && !isempty(args.node_checkpoint)
        # ── Retrain: load existing NODE and override time/tspan from data ──
        @info "Retraining NODE from checkpoint: $(args.node_checkpoint)"
        node, prev_args = load_node(args.node_checkpoint; verbose=true)
        node.tspan = tspan
        node.t = t
        # Optionally override solver tolerances from new args
        node.solver = args.solver
        node.abstol = args.abstol
        node.reltol = args.reltol
        @info "NODE reloaded" latent_dim = prev_args.latent_dim dense_mult = prev_args.dense_mult tspan = tspan solver = typeof(args.solver) reltol = args.reltol abstol = args.abstol t = t activation = args.activation
        baseline_eval = multi ? eval_node_loss_multi(node, zs, z0s; ts=ts) : eval_node_loss(node, z, z0)
        @info "Baseline eval before retraining" baseline_eval.mae baseline_eval.rmse baseline_eval.rel_l2

    else
        # ── Fresh training ──
        node = NODE(args.latent_dim, args.dense_mult; 
                    tspan=tspan, t=t, activation=args.activation, 
                    solver=args.solver, abstol=args.abstol, reltol=args.reltol)
        setup_lux!(node)
    end

    # Create ComponentArray on CPU BEFORE moving to GPU
    pinit = ComponentArray(node.p0)

    # ---- Move to GPU if requested ----
    @info "NODE training on device: $device"

    if multi
        zs  = [device(z_i) for z_i in zs]
        z0s = [device(z0_i) for z0_i in z0s]
        ts  = [device(t_i) for t_i in ts]
    else
        z    = device(z)
        z0   = device(z0)
        t    = device(t)
    end
    node.p0 = device(node.p0)
    node.st = device(node.st)

    # Move the ComponentArray to GPU
    pinit = device(pinit)

    # unified loss callable used by optimization
    loss_function = if multi
        x -> node_loss(args, node, zs, z0s; p=x, ts=ts)
    else
        x -> node_loss(args, node, z, z0; p=x)
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    # maxiters
    maxiters = args.maxiters
    if args.optimiser === OptimizationPolyalgorithms.PolyOpt
        maxiters *= 2
        @info "maxiters doubled for PolyOpt"
    end

    train_losses = Float64[]
    rmse_errors  = Float64[]
    test_losses  = Float64[]
    epochs       = Int[]
    eval_every   = 1  # change for sparser evaluation
    log_every    = 10  # print summary every N iterations

    # Callback function
    anim = Plots.Animation()
    iter_start_time = Ref(time())

    callback = function (state, l; plotting=false, gif=true)
        step = state.iter              # current iteration index
        if step % eval_every == 0
            push!(epochs, step)
            push!(train_losses, l)
            rmse_eval = multi ? eval_node_loss_multi(node, zs, z0s; p=state.u, ts=ts) : eval_node_loss(node, z, z0; p=state.u)
            push!(rmse_errors, rmse_eval.rmse)
            # Evaluate test loss with current params
            # test_l = L2_loss(node, z_test, z0_test; p=state.u, t=t_test)
            # push!(test_losses, test_l)
        end

        # Print summary every log_every iterations
        if step % log_every == 0 || step == 1
            elapsed = time() - iter_start_time[]
            avg_time = elapsed / step
            println(join([
                "Iter $(step)/$(maxiters)",
                "loss=$(round(l; digits=6))",
                "avg_time=$(round(avg_time; digits=3))s/iter",
                "elapsed=$(round(elapsed; digits=1))s",
                "ETA=$(round(avg_time * (maxiters - step); digits=1))s"
            ], " | "))
        end

        if gif
            if multi
                _, predss = loss_multiple_shoot_multi(node, zs, z0s; p=ComponentArray(state.u),
                    ts=ts, group_size=args.group_size, continuity_term=args.continuity_term)
                plt = plot_multiple_shoot_multi(node, predss, zs;
                    group_size=args.group_size, ts=ts, title_loss=l, n_reconstruct=args.n_reconstruct)
            elseif args.multiple_shooting
                _, preds = loss_multiple_shoot(node, z, z0; p=ComponentArray(state.u),
                                               t=node.t, group_size=args.group_size,
                                               continuity_term=args.continuity_term)
                plt = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=l, n_reconstruct=args.n_reconstruct)
            else
                plt = plot_node_trajectory(node, z, z0; p=state.u, loss=l)
            end
            frame(anim)
            plotting && display(plt)
        end
        return false
    end

    # actual training of NODE
    @info "Starting optimization" maxiters=maxiters optimiser=args.optimiser η=args.η
    iter_start_time[] = time()
    opt_instance = make_optimiser(args.optimiser, args.η)
    result = solve(optprob, opt_instance; callback=callback, maxiters=args.maxiters)
    total_time = time() - iter_start_time[]
    @info "Optimization finished" total_time=round(total_time; digits=1) final_loss=round(train_losses[end]; digits=6)

    final_eval = multi ? eval_node_loss_multi(node, zs, z0s; p=result.u, ts=ts) : eval_node_loss(node, z, z0; p=result.u)
    @info "Eval loss (comparable across runs)" final_eval.mae final_eval.rmse final_eval.rel_l2

    # Ensure final point is present in training curves when eval cadence skips it.
    if isempty(epochs) || epochs[end] != args.maxiters
        push!(epochs, args.maxiters)
        push!(train_losses, loss_function(result.u))
        push!(rmse_errors, final_eval.rmse)
    end

    # Move result back to CPU before saving
    cpu = cpu_device()
    node.p0 = cpu(result.u)

    # saving the model
    @info "Saving model and plots"
    timestamp = "NODE_" * Dates.format(now(), "udd-HHMMSS")

    # out_dir = joinpath("data", "NODE_models", timestamp)
    out_dir = joinpath(args.save_path, timestamp)
    mkpath(out_dir)

    # save optimized parameters
    node_path = joinpath(out_dir, "node_params.jld2")
    save_node(node_path, node, args)

    # final plot: match mode used during training
    final_loss = loss_function(node.p0)
    if multi
        _, predss = loss_multiple_shoot_multi(node, zs, z0s; p=node.p0, ts=ts,
            group_size=args.group_size, continuity_term=args.continuity_term)
        plt = plot_multiple_shoot_multi(node, predss, zs;
            group_size=args.group_size, ts=ts, title_loss=final_loss)
    elseif args.multiple_shooting
        _, preds = loss_multiple_shoot(node, z, z0; p=node.p0, t=node.t,
                                       group_size=args.group_size, continuity_term=args.continuity_term)
        plt = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=final_loss)
    else
        plt = plot_node_trajectory(node, z, z0; loss=final_loss)
    end
    png_path = joinpath(out_dir, "trajectories.pdf")
    @info "  Saved trajectory plot to $png_path"
    savefig(plt, png_path)

    metrics_path = joinpath(out_dir, "training_loss_rmse.pdf")
    if !isempty(epochs) && !isempty(train_losses) && !isempty(rmse_errors)
        loss_for_plot = max.(train_losses, eps(Float64))
        rmse_for_plot = max.(rmse_errors, eps(Float64))
        metrics_plot = plot(layout=grid(2, 1, heights=[0.5, 0.5]),
            size=(1000, 450),
            dpi=400,
            framestyle    = :box,
            gridalpha     = 0.20,
            gridlinewidth = 0.5,
            link          = :x,
            foreground_color_axis = :black,
            foreground_color_text = :black,
            left_margin   = 8Plots.mm,
            right_margin  = 6Plots.mm,
            top_margin    = 2Plots.mm,
            bottom_margin = 6Plots.mm,
        )

        log_decade_ticks(v) = begin
            lo, hi = floor(Int, log10(minimum(v))), ceil(Int, log10(maximum(v)))
            10.0 .^ (lo:hi)
        end

        plot!(metrics_plot[1], epochs, loss_for_plot;
            ylabel=L"\mathcal{L}", legend=false, yscale=:log10,
            yticks=log_decade_ticks(loss_for_plot),
            yminorticks=10, yminorgrid=true, minorgridalpha=0.05,
            linewidth=1.4, color=:steelblue,
            xformatter=_->"", xlabel="",
            bottom_margin=-2Plots.mm)
        plot!(metrics_plot[2], epochs, rmse_for_plot;
            xlabel="Iteration", ylabel="RMSE", legend=false, yscale=:log10,
            yticks=log_decade_ticks(rmse_for_plot),
            yminorticks=10, yminorgrid=true, minorgridalpha=0.10,
            linewidth=1.4, color=:firebrick,
            top_margin=-2Plots.mm)

        # final_loss = loss_for_plot[end]
        # final_rmse = rmse_for_plot[end]
        # last_epoch = epochs[end]
        # hline!(metrics_plot[1], [final_loss]; color=:steelblue, linestyle=:dot, alpha=0.6, label="")
        # hline!(metrics_plot[2], [final_rmse]; color=:firebrick, linestyle=:dot, alpha=0.6, label="")
        # annotate!(metrics_plot[1], last_epoch, final_loss,
        #     text(@sprintf("  final = %.3g", final_loss), :steelblue, :left, 10))
        # annotate!(metrics_plot[2], last_epoch, final_rmse,
        #     text(@sprintf("  final = %.3g", final_rmse), :firebrick, :left, 10))

        savefig(metrics_plot, metrics_path)
        @info "  Saved training metrics plot to $metrics_path"

        metrics_jld_path = joinpath(out_dir, "training_metrics.jld2")
        @save metrics_jld_path epochs train_losses rmse_errors
        @info "  Saved training metrics arrays to $metrics_jld_path"
    else
        @warn "Training metrics arrays are empty; skipping loss/RMSE plot"
    end

    gif_path = joinpath(out_dir, "training_trajectories.gif")
    try gif(anim, gif_path; fps = 15, show_msg=false)
        @info "  Saved training gif to $gif_path"
    catch e
        @warn "plotting turned off in callback, no gif saved"
    end
    
    if args.extrapolate && !multi
        extrapolation_plot, (z̃_train, z̃_test) = extrapolate_node(node_path)
        display(extrapolation_plot)
        extrapolation_path = joinpath(out_dir, "extrapolation_plot_loss.png")
        savefig(extrapolation_plot, extrapolation_path)
        @info "  Saved extrapolation plot to $extrapolation_path"
    end

    return node, node_path
end

# if abspath(PROGRAM_FILE) == @__FILE__
    # train_NODE(NodeArgs())
# end
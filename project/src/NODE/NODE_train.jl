make_optimiser(opt, η) = hasmethod(opt, Tuple{Float64}) ? opt(η) :
                         hasmethod(opt, Tuple{}) ? opt() :
                         error("Unsupported optimiser constructor: $(opt)")

function train_NODE(args::NodeArgs; 
    ae_bundle=nothing,
    normalizer=nothing, ae_args=nothing, kws...)

    device = args.use_gpu ? get_device() : cpu_device()

    if isnothing(ae_bundle)
        # Original path: load pre-saved latent data from disk
        z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
    else
        # New path: encode on-the-fly using the trained AE already in memory
        @info "Encoding latent vectors from AE in memory (no disk I/O)"
        z, t, tspan, z0 = get_latent_vectors(ae_bundle, normalizer, ae_args; device=device, downsample=args.downsample)
        # ae, ae_ps, ae_st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st

        # Optionally downsample
        if args.downsample > 0 && args.downsample < size(z, 2)
            idx = downsample_equal(collect(1:size(z, 2)), args.downsample)
            z = z[:, idx]
            t = t[idx]
            tspan = (t[1], t[end])
            z0 = z[:, 1]
        end
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
        baseline_eval = eval_node_loss(node, z, z0)
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

    z    = device(z)
    z0   = device(z0)
    t    = device(t)
    node.p0 = device(node.p0)
    node.st = device(node.st)

    # Move the ComponentArray to GPU
    pinit = device(pinit)

    # unified loss callable used by optimization
    @inline loss_function(x) = node_loss(args, node, z, z0; p=x)

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

    callback = function (state, l; plotting=false, gif=false)
        step = state.iter              # current iteration index
        if step % eval_every == 0
            push!(epochs, step)
            push!(train_losses, l)
            rmse_eval = eval_node_loss(node, z, z0; p=state.u)
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
            if args.multiple_shooting
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

    final_eval = eval_node_loss(node, z, z0; p=result.u)
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
    timestamp = "NODE_" * Dates.format(now(), "udd-HHMM")

    # out_dir = joinpath("data", "NODE_models", timestamp)
    out_dir = joinpath(args.save_path, timestamp)
    mkpath(out_dir)

    # save optimized parameters
    node_path = joinpath(out_dir, "node_params.jld2")
    save_node(node_path, node, args)

    # final plot: match mode used during training
    final_loss = loss_function(node.p0)
    if args.multiple_shooting
        _, preds = loss_multiple_shoot(node, z, z0; p=node.p0, t=node.t,
                                       group_size=args.group_size, continuity_term=args.continuity_term)
        plt = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=final_loss)
    else
        plt = plot_node_trajectory(node, z, z0; loss=final_loss)
    end
    png_path = joinpath(out_dir, "trajectories.png")
    @info "  Saved trajectory plot to $png_path"
    savefig(plt, png_path)

    metrics_path = joinpath(out_dir, "training_loss_rmse.png")
    if !isempty(epochs) && !isempty(train_losses) && !isempty(rmse_errors)
        loss_for_plot = max.(train_losses, eps(Float64))
        rmse_for_plot = max.(rmse_errors, eps(Float64))
        metrics_plot = plot(layout=(2, 1), size=(900, 700))
        plot!(metrics_plot[1], epochs, loss_for_plot;
            xlabel="Iteration", ylabel="Loss", title="Training Loss", legend=false, yscale=:log10)
        plot!(metrics_plot[2], epochs, rmse_for_plot;
            xlabel="Iteration", ylabel="RMSE", title="Training RMSE", legend=false, color=:red, yscale=:log10)
        savefig(metrics_plot, metrics_path)
        @info "  Saved training metrics plot to $metrics_path"
    else
        @warn "Training metrics arrays are empty; skipping loss/RMSE plot"
    end

    gif_path = joinpath(out_dir, "training_trajectories.gif")
    try gif(anim, gif_path; fps = 15, show_msg=false)
        @info "  Saved training gif to $gif_path"
    catch e
        @warn "plotting turned off in callback, no gif saved"
    end
    
    if args.extrapolate
        extrapolation_plot, (ẑ_train, ẑ_test) = extrapolate_node(node_path)
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
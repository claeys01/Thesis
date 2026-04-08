make_optimiser(opt, η) = hasmethod(opt, Tuple{Float64}) ? opt(η) :
                         hasmethod(opt, Tuple{}) ? opt() :
                         error("Unsupported optimiser constructor: $(opt)")

function train_NODE(args; kws...)
    z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
    
    # z_test, t_test, tspan_test, z0_test = get_NODE_data(args.test_latent_path; downsample=args.test_downsample)

    node = NODE(args.latent_dim, args.dense_mult; 
                tspan=tspan, t=t, activation=args.activation, 
                solver=args.solver, abstol=args.abstol, reltol=args.reltol)
    setup_lux!(node)

    # ---- Move to GPU if requested ----
    device = get_device()
    @info "NODE training on device: $device"

    z    = device(z)
    z0   = device(z0)
    t    = device(t)
    node.p0 = device(node.p0)
    node.st = device(node.st)

    # unified loss callable used by optimization
    @inline loss_function(x) = node_loss(args, node, z, z0; p=x)

    pinit = ComponentArray(node.p0)
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
    test_losses  = Float64[]
    epochs       = Int[]
    eval_every   = 1  # change for sparser evaluation
    log_every    = 10  # print summary every N iterations

    # Callback function
    anim = Plots.Animation()
    iter_start_time = Ref(time())

    callback = function (state, l; plotting=false)
        step = state.iter              # current iteration index
        if step % eval_every == 0
            push!(epochs, step)
            push!(train_losses, l)
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

        if plotting
            if args.multiple_shooting
                _, preds = loss_multiple_shoot(node, z, z0; p=ComponentArray(state.u),
                                               t=node.t, group_size=args.group_size,
                                               continuity_term=args.continuity_term)
                plt = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=l, n_reconstruct=args.n_reconstruct)
            else
                plt = plot_node_trajectory(node, z, z0; p=state.u, loss=l)
            end
            frame(anim); display(plt)
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

    node.p0 = result.u # set final network parameters in struct

    # saving the model
    @info "Saving model and plots"
    timestamp = Dates.format(now(), "udd-HHMM")

    out_dir = joinpath("data", "NODE_models", timestamp)
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
end

# if abspath(PROGRAM_FILE) == @__FILE__
    # train_NODE(NodeArgs())
# end
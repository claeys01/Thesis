make_optimiser(opt, η) = hasmethod(opt, Tuple{Float64}) ? opt(η) :
                         hasmethod(opt, Tuple{}) ? opt() :
                         error("Unsupported optimiser constructor: $(opt)")

function train_NODE(args; kws...)
    z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)
    
    z_test, t_test, tspan_test, z0_test = get_NODE_data(args.test_latent_path; downsample=args.test_downsample)

    node = NODE(args.latent_dim, args.dense_mult; 
                tspan=tspan, t=t, activation=args.activation, 
                solver=args.solver, abstol=args.abstol, reltol=args.reltol)
    setup_lux!(node)

    # unified loss callable used by optimization
    @inline loss_function(x) = node_loss(args, node, z, z0; p=x)

    pinit = ComponentArray(node.p0)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    # progress bar used by callback (created with known maxiters)
    maxiters = args.maxiters
    if args.optimiser === OptimizationPolyalgorithms.PolyOpt
        maxiters *= 2
        @info "maxiters doubled for PolyOpt"
    end
    pb = Progress(maxiters; desc="Optimizing NODE")

    train_losses = Float64[]
    test_losses  = Float64[]
    epochs       = Int[]
    eval_every   = 1  # change for sparser evaluation

    # Callback function
    anim = Plots.Animation()
    callback = function (state, l; plotting=true)
        step = state.iter              # current iteration index
        if step % eval_every == 0
            # push!(epochs, step)
            # push!(train_losses, L2_loss(node, z, z0; p=state.u))
            # push!(train_losses, l)
            # Evaluate test loss with current params
            # test_l = L2_loss(node, z_test, z0_test; p=state.u, t=t_test)
            # push!(test_losses, test_l)
            # push!(test_losses, node_loss(args, node, z_test, z0_test; p=state.u, t=t_test))
        end
        if plotting
            # title = "Iteration $step, loss = "
            if args.multiple_shooting
                # compute current segment predictions for visualization

                _, preds = loss_multiple_shoot(node, z, z0; p=ComponentArray(state.u),
                                               t=node.t, group_size=args.group_size,
                                               continuity_term=args.continuity_term)
                plt = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=l, n_reconstruct=args.n_reconstruct)
            else
                plt = plot_node_trajectory(node, z, z0; p=state.u, loss=l)
            end
            frame(anim); display(plt)
        end
        next!(pb; showvalues=[(:step, step) (:loss, l)])
        return false
    end

    # actual training of NODE
    @info "Starting optimization"
    opt_instance = make_optimiser(args.optimiser, args.η)
    result = solve(optprob, opt_instance; callback=callback, maxiters=args.maxiters)
    finish!(pb)
    @info "Optimization finished"

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


    # # Loss curve plot (train vs test)
    # plt_loss = plot(epochs, train_losses; label="train", xlabel="epoch", ylabel="loss",
    #                 title="NODE train/test loss", yscale=:log10, linewidth=2)
    # if length(test_losses) == length(epochs)
    #     plot!(plt_loss, epochs, test_losses; label="test", linewidth=2)
    # end
    # loss_path = joinpath(out_dir, "loss_curve.png")
    # savefig(plt_loss, loss_path)
    # @info "  Saved loss curve to $loss_path"

    gif_path = joinpath(out_dir, "training_trajectories.gif")
    try gif(anim, gif_path; fps = 15, show_msg=false)
        @info "  Saved training gif to $gif_path"
    catch e
        @warn "plotting turned off in callback, no gif saved"
    end
    
    extrapolation_plot, (ẑ_train, ẑ_test) = extrapolate_node(node_path)
    display(extrapolation_plot)
    extrapolation_path = joinpath(out_dir, "extrapolation_plot_loss.png")
    savefig(extrapolation_plot, extrapolation_path)
    @info "  Saved extrapolation plot to $extrapolation_path"

    # Save predictions
    # preds_path = joinpath(out_dir, "predictions.jld2")
    # @save preds_path ẑ_train ẑ_test
    # @info "  Saved predictions to $preds_path"
end

# if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
#     train_NODE(NodeArgs())
# end
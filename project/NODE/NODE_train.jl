using Revise
using NNlib
using Optimization, OptimizationOptimisers
using Dates
using ProgressMeter
using OptimizationPolyalgorithms




includet("NODE_core.jl")


function train_NODE(args; kws...)

    z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)

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
    pb = Progress(args.maxiters; desc="Optimizing NODE")
    cb_step = Ref(0)

    # Callback function
    anim = Plots.Animation()
    callback = function (state, l; plotting=true)
        cb_step[] += 1
        if plotting
            if args.multiple_shooting
                # compute current segment predictions for visualization
                _, preds = loss_multiple_shoot(node, z, z0; p=ComponentArray(state.u),
                                               t=node.t, group_size=args.group_size,
                                               continuity_term=args.continuity_term)
                p = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=l, n_reconstruct=args.n_reconstruct)
            else
                p = plot_node_trajectory(node, z, z0; p=state.u, loss=l)
            end
            frame(anim)
            display(p)
        end
        next!(pb; showvalues=[(:step, cb_step[]) (:loss, l)])
        return false
    end

    # actual training of NODE
    # result_neuralode = solve(optprob, args.optimiser(args.η); callback = callback, maxiters = args.maxiters)
    result_neuralode = solve(optprob, PolyOpt(); callback = callback, maxiters = args.maxiters)

    finish!(pb)


     
    node.p0 = result_neuralode.u # set final network parameters in struct

    # saving the model
    out_dir = joinpath("data", "NODE_models", Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))

    mkpath(out_dir)
    

    # save optimized parameters
    node_path = joinpath(out_dir, "node_params.jld2")
    save_node(node_path, node, args)

    # final plot: match mode used during training
    final_loss = loss_function(node.p0)
    if args.multiple_shooting
        loss, preds = loss_multiple_shoot(node, z, z0; p=node.p0, t=node.t,
                                       group_size=args.group_size, continuity_term=args.continuity_term)
        @show final_loss, loss
        p = plot_multiple_shoot(node, preds, z; group_size=args.group_size, title_loss=final_loss)
    else
        p = plot_node_trajectory(node, z, z0; loss=final_loss)
    end
    png_path = joinpath(out_dir, "trajectories.png")
    @info "Saved trajectory plot to $png_path"
    savefig(p, png_path)

    gif_path = joinpath(out_dir, "training_trajectories.gif")
    gif(anim, gif_path; fps = 15)
    @info "Saved training gif to $gif_path"
    nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    train_NODE(NodeArgs())
end
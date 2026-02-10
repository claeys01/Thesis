using Revise
using NNlib
using Optimization, OptimizationOptimisers
using Dates
using ProgressMeter



includet("NODE_core.jl")


function train_MS_NODE(args; kws...)

    z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)

    node = NODE(args.latent_dim, args.dense_mult; 
                tspan=tspan, t=t, activation=args.activation, 
                solver=args.solver, abstol=args.abstol, reltol=args.reltol)
    setup_lux!(node)

    function loss_function(data, pred)
        return sum(abs2, data - pred)
    end
    
    pinit = ComponentArray(node.p0)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    # progress bar used by callback (created with known maxiters)
    pb = Progress(args.maxiters; desc="Optimizing NODE")
    cb_step = Ref(0)

    # Callback function
    callback = function (state, l; plotting=true)
        cb_step[] += 1
        if plotting
            p = plot_node_trajectory(node, z, z0; p=state.u, loss=l)
            display(p)
        end
        next!(pb; showvalues=[(:step, cb_step[]) (:loss, l)])
        return false
    end

    # actual training of NODE
    result_neuralode = solve(optprob, args.optimiser(args.η); callback = callback, maxiters = args.maxiters)

    finish!(pb)
     
    node.p0 = result_neuralode.u # set final network parameters in struct

    # saving the model
    out_dir = joinpath("data", "NODE_models", Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
    mkpath(out_dir)
    # save optimized parameters
    node_path = joinpath(out_dir, "node_params.jld2")
    save_node(node_path, node, args)

    p = plot_node_trajectory(node, z, z0)
    png_path = joinpath(out_dir, "trajectories.png")
    @info "Saved trajectory plot to $png_path"

    savefig(p, png_path)

    nothing
end

if abspath(PROGRAM_FILE) == (@__FILE__) || isinteractive()
    train_MS_NODE(NodeArgs())
end
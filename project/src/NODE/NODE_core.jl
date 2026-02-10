
Base.@kwdef mutable struct NodeArgs
    η = 0.01                    # learning rate
    # optimiser = OptimizationPolyalgorithms.PolyOpt #PolyOpt
    optimiser = OptimizationOptimisers.Adam
    maxiters = 250
    solver = Tsit5()
    reltol = 1e-3
    abstol = 1e-5
    seed = 42                   # random seed
    latent_dim = 16
    dense_mult = 2
    activation = tanhshrink
    n_reconstruct = 4
    downsample = 300
    test_downsample = 500
    clip_bc = true
    use_gpu = false             # use GPU
    multiple_shooting = true
    group_size = 20
    continuity_term = 200
    save_path = "data/models/NODE_models"   # results dir
    train_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_train.jld2"
    test_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_test.jld2"
    total_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent.jld2"
end

function get_NODE_data(latent_path; downsample=-1, verbose=true)
    @load latent_path latent_data
    n_total = size(latent_data.z, 2)
    idx = collect(1:n_total)
    idx_downsample = downsample_equal(idx, downsample)
    z = latent_data.z[:, idx_downsample]
    t = latent_data.time[idx_downsample]
    tspan = (t[1], t[end])
    z0 = z[:, 1]
    verbose && @info "Instantiated latent NODE data" size(z) size(t) tspan size(z0)
    return z, t, tspan, z0
end

mutable struct NODE
    dudt::Any
    tspan::Tuple{<:Real,<:Real}
    solver::Any
    abstol::Float64
    reltol::Float64
    t::Union{AbstractVector{<:Real},Nothing}
    p0::Any # network parameters
    st::Any # network state
end

# constructor 
function NODE(latent_dim, dense_mult; tspan=(0.0f0, 1.0f0), solver=Tsit5(), abstol=1e-6, reltol=1e-6, t=nothing, activation=tanh, verbose=true)
    hidden_nodes = dense_mult * latent_dim
    nn = Chain(
        Dense(latent_dim, hidden_nodes, activation),
        # Dense(hidden_nodes, hidden_nodes, activation),
        Dense(hidden_nodes, latent_dim))
    verbose && @info "NODE initialized" latent_dim = latent_dim dense_mult = dense_mult tspan = tspan solver = typeof(solver) reltol = reltol abstol = abstol t = t activation = activation
    return NODE(nn, tspan, solver, abstol, reltol, t, nothing, nothing)
end


function setup_lux!(node::NODE; rng=Xoshiro(0), verbose=true)
    ps, st = Lux.setup(rng, node.dudt)
    node.p0 = ps
    node.st = st
    verbose && @info "NODE weights initialized"
    return node
end

# Build a NeuralODE from the (possibly reconstructed) model and solve for given initial state z0 and params p.
function predict(node::NODE, z0; p=nothing, t=nothing)
    t = t === nothing ? node.t : t
    tspan = t === nothing ? node.tspan : (t[1], t[end])
    p_used = p === nothing ? node.p0 : p
    nnode = NeuralODE(node.dudt, tspan, node.solver; saveat=t, abstol=node.abstol, reltol=node.reltol)
    sol = nnode(z0, p_used, node.st)

    if isa(sol, Tuple)
        sol = sol[1]
    end

    return sol
end

# Convenience: produce a (latent_dim, n_timepoints) Array prediction 
function predict_array(node::NODE, z0; p=nothing, t=nothing)
    sol = predict(node, z0; p=p, t=t)
    return Array(sol)
end

# Mean-squared error loss between data `z` and NODE prediction (z shaped like (latent_dim, n_t))
function L2_loss(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing)
    pred = predict_array(node, z0; p=p, t=t)
    @assert size(pred) == size(z) "prediction size $(size(pred)) != data size $(size(z))"
    return sum(abs2, z .- pred)
end

# small utility to print a short summary
function Base.show(io::IO, node::NODE)
    println(io, "NODE wrapper:")
    println(io, "  dudt = $(typeof(node.dudt))")
    println(io, "  tspan = $(node.tspan)")
    println(io, "  solver = $(node.solver)")
    println(io, "  t = $(node.t)")
    if node.p0 !== nothing
        println(io, "  p0 length = $(length(node.p0))")
    end
end

function build_node_problem(node::NODE, z0)
    # Convert the Lux model to a function matching (u,p,t) -> du
    dudt(u, p, t) = node.dudt(u, p, node.st)[1]
    ODEProblem(dudt, z0, node.tspan, ComponentArray(node.p0))
end

function loss_multiple_shoot(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing,
    group_size::Int=20, continuity_term::Real=200)
    tsteps = t === nothing ? node.t : t
    @assert tsteps !== nothing "t must be specified or set in node for multiple shooting"

    prob_node = build_node_problem(node, z0)

    # map parameters into ComponentArray to preserve axes
    p_used = p === nothing ? node.p0 : p

    # simple L2 loss over all predicted segments (sum of segment losses)
    seg_loss(data_seg, pred_seg) = sum(abs, data_seg .- pred_seg)

    # l: scalar loss; preds: Vector of predicted segment matrices (latent_dim × segment_len)
    l, preds = DiffEqFlux.multiple_shoot(p_used, z, tsteps, prob_node, seg_loss,
        node.solver, group_size; continuity_term=continuity_term)
    return l, preds
end

function node_loss(args::NodeArgs, node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing)
    if args.multiple_shooting
        l, _ = loss_multiple_shoot(node, z, z0; p=p, t=t,
            group_size=args.group_size, continuity_term=args.continuity_term)
        return l
    else
        return L2_loss(node, z, z0; p=p, t=t)
    end
end

function plot_multiple_shoot(node::NODE, preds::Vector{<:AbstractMatrix}, z::AbstractMatrix;
    group_size::Int, title_loss=nothing, n_reconstruct=4)
    ranges = DiffEqFlux.group_ranges(size(z, 2), group_size)
    p = plot()
    if !isnothing(title_loss)
        title!(p, "loss = $(title_loss)")
    end

    idx_samples = round.(Int, range(1, stop=size(z, 1), length=n_reconstruct))

    # cycle colors to avoid bounds errors
    palette = [:black, :red, :blue, :green, :purple, :orange, :yellow]
    ncolors = length(palette)

    for (sidx, lat_idx) in enumerate(idx_samples)
        c = palette[(sidx - 1) % ncolors + 1]
        for (j, rg) in enumerate(ranges)
            # plot data and prediction for the selected latent component
            plot!(p, node.t[rg], z[lat_idx, rg];
                  color=c, linestyle=:solid, label=sidx == 1 && j == 1 ? "z (truth)" : "")
            plot!(p, node.t[rg], preds[j][lat_idx, :];
                  color=c, linestyle=:dash, label=sidx == 1 && j == 1 ? "ẑ (pred)" : "")

            # start/end markers for the segment to visualize overlaps
            t_start, t_end = node.t[rg][1], node.t[rg][end]
            z_start, z_end = z[lat_idx, rg][1], z[lat_idx, rg][end]
            pred_start, pred_end = preds[j][lat_idx, 1], preds[j][lat_idx, end]

            # prediction markers (open circle at start, square at end)
            scatter!(p, [t_start], [pred_start]; color=c, marker=:x, markersize=6, markerstrokecolor=c,
                     markerstrokewidth=1.5, fillalpha=0.0,
                     label=sidx == 1 && j == 1 ? "pred start/end" : nothing)
            scatter!(p, [t_end], [pred_end]; color=c, marker=:x, markersize=6, markerstrokecolor=c,
                     markerstrokewidth=1.5, fillalpha=0.0,
                     label=nothing)
        end
    end
    return p
end

function plot_node_trajectory(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing, n_reconstruct=4, loss=nothing, plt=nothing, labels=true)
    idx_samples = round.(Int, range(1, stop=size(z, 1), length=n_reconstruct))
    z_samples = [vec(z[i, :]) for i in idx_samples]  # Vector of 8 one-dimensional arrays, each length 179
    ẑ = predict_array(node, z0; p=p, t=t)
    ẑ_samples = [vec(ẑ[i, :]) for i in idx_samples]  # Vector of 4 one-dimensional arrays, each length 179
    fig = plt === nothing ? plot() : plt
    if !isnothing(loss)
        title!(fig, "loss = $(loss)")
    end
    colors = [:black, :red, :blue, :green, :purple, :orange, :yellow]

    lw_truth = 2.2
    lw_pred  = 1.4
    α_truth  = 0.95
    α_pred   = 0.70
    
    for i in 1:n_reconstruct
        c = colors[i]
        plot!(fig, node.t, z_samples[i];
            color=c, linewidth=lw_truth, alpha=α_truth,
            label = labels ? "z $(idx_samples[i]) (truth)" : ""
        )
        plot!(fig, node.t, ẑ_samples[i];
            color=c, linewidth=lw_pred, alpha=α_pred, linestyle=:dash,
            label = labels ? "ẑ $(idx_samples[i]) (NODE)" : ""
        )
    end
    return fig
end

function save_node(path::AbstractString, node::NODE, args::NodeArgs)
    args_copy = deepcopy(args)
    node_tspan = node.tspan
    node_t = node.t
    @save path node_p0 = node.p0 node_st = node.st node_args = args_copy node_tspan node_t
    @info "  Saved NODE to $path"
end

function load_node(path::AbstractString; verbose=true)
    @load path node_p0 node_st node_args node_tspan node_t
    node = NODE(node_args.latent_dim, node_args.dense_mult; activation=node_args.activation, verbose=false)
    setup_lux!(node, verbose=false)   # create a matching structure
    node.p0 = node_p0
    node.st = node_st
    node.tspan = node_tspan
    node.t = node_t
    verbose && @info "NODE sucessfully loaded from $path"
    return node, node_args
end


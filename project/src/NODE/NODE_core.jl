Base.@kwdef mutable struct NodeArgs
    η = 0.01                    # learning rate
    # optimiser = OptimizationPolyalgorithms.PolyOpt #PolyOpt
    optimiser = OptimizationOptimisers.AdamW
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
    retrain = false
    node_checkpoint = ""
    extrapolate = true
    group_size = 20
    continuity_term = 200
    save_path = "data/models/NODE_models"   # results dir
    train_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_train.jld2"
    test_latent_path =  "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000_test.jld2"
    total_latent_path = "data/latent_data/16/RE2500/2e8/U_128_latent_curldiv_E1000.jld2"
end

function get_NODE_data(latent_path; downsample=-1, verbose=true)
    @load latent_path latent_data
    n_total = size(latent_data.z, 2)
    idx = collect(1:n_total)
    idx_downsample = downsample <= 0 || downsample >= n_total ? idx : downsample_equal(idx, downsample)
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
        # Dense(hidden_nodes, 2*hidden_nodes, activation),
        # Dense(2*hidden_nodes, hidden_nodes, activation),
        Dense(hidden_nodes, latent_dim))
    verbose && @info "NODE initialized" latent_dim = latent_dim dense_mult = dense_mult tspan = tspan solver = typeof(solver) reltol = reltol abstol = abstol t = typeof(t) activation = activation
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

    # if isa(sol, Tuple)
    #     sol = sol[1]
    # end
    # @show sol.stats.nf
    return sol
end

# Convenience: produce a (latent_dim, n_timepoints) Array prediction 
function predict_array(node::NODE, z0; p=nothing, t=nothing, onlysol=true)
    sol = predict(node, z0; p=p, t=t)
    onlysol ? Array(sol[1]) : sol
    # return Array(sol)
end

# Mean-squared error loss between data `z` and NODE prediction (z shaped like (latent_dim, n_t))
function L2_loss(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing)
    pred = predict_array(node, z0; p=p, t=t)
    @assert size(pred) == size(z) "prediction size $(size(pred)) != data size $(size(z))"
    return sum(abs2, z .- pred)
end

function eval_node_loss(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing)
    pred = predict_array(node, z0; p=p, t=t)
    mae = mean(abs, z .- pred)
    rmse = sqrt(mean(abs2, z .- pred))
    rel_l2 = sqrt(sum(abs2, z .- pred)) / sqrt(sum(abs2, z))
    return (; mae, rmse, rel_l2)
end

# Aggregate eval over multiple disjoint trajectories.
function eval_node_loss_multi(node::NODE, zs::Vector{<:AbstractMatrix}, z0s::Vector;
        p=nothing, ts::Vector)
    sse, n_pts, sse_z = 0.0, 0, 0.0
    abs_sum = 0.0
    for i in eachindex(zs)
        pred = predict_array(node, z0s[i]; p=p, t=ts[i])
        size(pred) == size(zs[i]) || (@warn "eval_multi size mismatch on chunk $i" pred=size(pred) z=size(zs[i]); continue)
        diff = zs[i] .- pred
        sse += sum(abs2, diff)
        abs_sum += sum(abs, diff)
        n_pts += length(diff)
        sse_z += sum(abs2, zs[i])
    end
    n_pts == 0 && return (; mae=NaN, rmse=NaN, rel_l2=NaN)
    return (; mae=abs_sum/n_pts, rmse=sqrt(sse/n_pts), rel_l2=sqrt(sse)/sqrt(sse_z))
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

function build_node_problem(node::NODE, z0; p=nothing, tspan=nothing)
    p_used = p === nothing ? node.p0 : p
    tspan_used = tspan === nothing ? node.tspan : tspan
    # Convert the Lux model to a function matching (u,p,t) -> du
    dudt(u, p, t) = node.dudt(u, p, node.st)[1]
    ODEProblem(dudt, z0, tspan_used, p_used)
end

seg_loss(data_seg, pred_seg) = sum(abs2, data_seg .- pred_seg)

function loss_multiple_shoot(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing,
    group_size::Int=20, continuity_term::Real=200)
    tsteps = t === nothing ? node.t : t
    @assert tsteps !== nothing "t must be specified or set in node for multiple shooting"

    # map parameters into ComponentArray to preserve axes
    p_used = p === nothing ? node.p0 : p

    prob_node = build_node_problem(node, z0; p=p_used)

    # simple L2 loss over all predicted segments (sum of segment losses)

    # l: scalar loss; preds: Vector of predicted segment matrices (latent_dim × segment_len)
    l, preds = DiffEqFlux.multiple_shoot(p_used, z, tsteps, prob_node, seg_loss,
        node.solver, group_size; continuity_term=continuity_term)
    return l, preds
end

# Multi-trajectory multiple shooting: each chunk gets its own ODEProblem with its own tspan, but with same parameter weights
function loss_multiple_shoot_multi(node::NODE, zs::Vector{<:AbstractMatrix}, z0s::Vector;
        p=nothing, ts::Vector, group_size::Int=20, continuity_term::Real=200)
    p_used = p === nothing ? node.p0 : p

    @assert all(size(z, 2) >= group_size for z in zs) "all chunks must have >= group_size points"

    results = map(eachindex(zs)) do i
        prob_i = build_node_problem(node, z0s[i]; p=p_used, tspan=(ts[i][1], ts[i][end]))
        DiffEqFlux.multiple_shoot(p_used, zs[i], ts[i], prob_i, seg_loss,
            node.solver, group_size; continuity_term=continuity_term)
    end

    total_loss = sum(r -> r[1], results)
    preds_all = [r[2] for r in results]
    return total_loss, preds_all
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

# Multi-trajectory dispatch: always uses multiple shooting, summed across chunks.
function node_loss(args::NodeArgs, node::NODE, zs::Vector{<:AbstractMatrix}, z0s::Vector;
        p=nothing, ts::Vector)
    l, _ = loss_multiple_shoot_multi(node, zs, z0s; p=p, ts=ts,
        group_size=args.group_size, continuity_term=args.continuity_term)
    return l
end

function plot_multiple_shoot(node::NODE, preds::Vector{<:AbstractMatrix}, z::AbstractMatrix;
    group_size::Int, title_loss=nothing, n_reconstruct=4, t=nothing)
    ts = t === nothing ? node.t : t
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
            plot!(p, ts[rg], z[lat_idx, rg];
                  color=c, linestyle=:solid, label=sidx == 1 && j == 1 ? "z (truth)" : "")
            plot!(p, ts[rg], preds[j][lat_idx, :];
                  color=c, linestyle=:dash, label=sidx == 1 && j == 1 ? "ẑ (pred)" : "")

            # start/end markers for the segment to visualize overlaps
            t_start, t_end = ts[rg][1], ts[rg][end]
            z_start, z_end = z[lat_idx, rg][1], z[lat_idx, rg][end]
            pred_start, pred_end = preds[j][lat_idx, 1], preds[j][lat_idx, end]

            # prediction markers (open circle at start, square at end)
            scatter!(p, [t_start], [pred_start]; color=c, marker=:vline, markersize=6, markerstrokecolor=c,
                     markerstrokewidth=1.5, fillalpha=0.0,
                     label=sidx == 1 && j == 1 ? "pred start/end" : nothing)
            scatter!(p, [t_end], [pred_end]; color=c, marker=:vline, markersize=6, markerstrokecolor=c,
                     markerstrokewidth=1.5, fillalpha=0.0,
                     label=nothing)
        end
    end
    return p
end

# Overlay all trajectories on a single plot so chunks visibly extend the same data.
function plot_multiple_shoot_multi(node::NODE, predss::Vector, zs::Vector{<:AbstractMatrix};
        group_size::Int, ts::Vector, title_loss=nothing, n_reconstruct=4)
    n = length(zs)
    p = plot(; size=(1100, 500))
    if title_loss !== nothing
        title!(p, "$n chunks (total loss = $(title_loss))")
    end

    latent_dim = size(zs[1], 1)
    idx_samples = round.(Int, range(1, stop=latent_dim, length=n_reconstruct))
    palette = [:black, :red, :blue, :green, :purple, :orange, :yellow]
    ncolors = length(palette)

    for i in 1:n
        z = zs[i]
        preds = predss[i]
        t = ts[i]
        ranges = DiffEqFlux.group_ranges(size(z, 2), group_size)

        for (sidx, lat_idx) in enumerate(idx_samples)
            c = palette[(sidx - 1) % ncolors + 1]
            for (j, rg) in enumerate(ranges)
                first_chunk = (i == 1 && j == 1 && sidx == 1)
                plot!(p, t[rg], z[lat_idx, rg];
                      color=c, linestyle=:solid, alpha=0.9,
                      label=first_chunk ? "z (truth)" : "")
                plot!(p, t[rg], preds[j][lat_idx, :];
                      color=c, linestyle=:dash, alpha=0.9,
                      label=first_chunk ? "ẑ (pred)" : "")

                t_start, t_end = t[rg][1], t[rg][end]
                pred_start, pred_end = preds[j][lat_idx, 1], preds[j][lat_idx, end]
                scatter!(p, [t_start, t_end], [pred_start, pred_end];
                         color=c, marker=:vline, markersize=5, markerstrokecolor=c,
                         markerstrokewidth=1.5, fillalpha=0.0,
                         label=first_chunk ? "pred start/end" : nothing)
            end

            # mark chunk boundary on the time axis (once per chunk, on first latent comp)
            if sidx == 1
                vline!(p, [t[1]]; color=:gray, linestyle=:dot, alpha=0.4,
                       label=(i == 1 ? "chunk start" : ""))
            end
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

"""
    get_latent_vectors(ae, ps, st, normalizer, ae_args; device=cpu_device())

Encode simulation data into latent vectors using a trained AE already in memory.
Returns `(z, t, tspan, z0)` on CPU.
"""
function get_latent_vectors(ae_bundle, normalizer::Normalizer, ae_args::LuxArgs; downsample=300, device=cpu_device())
    ae, ps, st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata; verbose=true)
    
    train_idx = get_trainval_idx(simdata, ae_args.t_training, downsample)
    
    x_in, _, _ = build_batch(
        EpochData(get_data_in(simdata.u, simdata.μ₀; idx=train_idx)...), 
        1:downsample; normalizer=normalizer
    )
    
    x_in = device(x_in)
    z, _ = ae.encoder(x_in, device(ps.encoder), device(LuxCore.testmode(st.encoder)))
    x_in = nothing; GC.gc()
    # Always return CPU arrays for NODE training
    z = Array(cpu_device()(z))
    t = simdata.time[train_idx]
    tspan = (t[1], t[end])
    z0 = z[:, 1]
    simdata = nothing; GC.gc()
    @info "Generated latent vectors" size(z) n_samples=length(train_idx) time_range=tspan
    return z, t, tspan, z0
end


function get_latent_chunks(ae_bundle, normalizer::Normalizer, ae_args::LuxArgs;
        downsample=300, device=cpu_device(), min_chunk_size::Int=21)
    ae, ps, st = ae_bundle.ae, ae_bundle.ps, ae_bundle.st
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata; verbose=true)

    chunks = simdata.chunk_ranges
    #     !isempty(simdata.chunk_ranges) ? simdata.chunk_ranges : [1:length(simdata.time)]
    # catch
    #     [1:length(simdata.time)]
    # end
    total_len = sum(length, chunks)

    zs = Vector{Matrix{Float32}}()
    ts = Vector{Vector{Float32}}()
    tspans = Vector{Tuple{Float32,Float32}}()
    z0s = Vector{Vector{Float32}}()

    for (i, chunk_rg) in enumerate(chunks)
        local_idx_full = i == 1 ? [j for j in chunk_rg if simdata.time[j] < ae_args.t_training] : collect(chunk_rg)
        if length(local_idx_full) < min_chunk_size
            @warn "Chunk $i has $(length(local_idx_full)) points (< min_chunk_size=$min_chunk_size) — skipping"
            continue
        end

        # downsample the chunks relative to their size of the total simdata
        n_i = max(min_chunk_size, round(Int, downsample * length(local_idx_full) / total_len))
        n_i = clamp(n_i, min_chunk_size, length(local_idx_full))
        idx_i = downsample_equal(local_idx_full, n_i)

        # encode the flow snapshot to the latent space
        x_in, _, _ = build_batch(
            EpochData(get_data_in(simdata.u, simdata.μ₀; idx=idx_i)...),
            1:length(idx_i); normalizer=normalizer
        )
        x_in = device(x_in)
        z_i_dev, _ = ae.encoder(x_in, device(ps.encoder), device(LuxCore.testmode(st.encoder)))
        x_in = nothing; GC.gc()
        z_i = Array(cpu_device()(z_i_dev))
        t_i = simdata.time[idx_i]

        push!(zs, z_i)
        push!(ts, t_i)
        push!(tspans, (t_i[1], t_i[end]))
        push!(z0s, z_i[:, 1])
    end

    simdata = nothing; GC.gc()
    @info "Generated multi-trajectory latent vectors" n_chunks=length(zs) sizes=[size(z) for z in zs] tspans=tspans
    return zs, ts, tspans, z0s
end


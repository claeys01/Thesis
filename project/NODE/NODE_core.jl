using DiffEqFlux
using OrdinaryDiffEq
using Lux
using Random
using ComponentArrays
using JLD2
using NNlib
using Optimization, OptimizationOptimisers
using Plots


includet("../custom.jl")

Base.@kwdef mutable struct NodeArgs
    η = 0.05                    # learning rate
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
    downsample = -1
    clip_bc = true
    use_gpu = false             # use GPU
    save_path = "data/models/NODE_models"   # results dir
    period_latent_path = "data/latent_data/16/RE2500/U_128_latent_period.jld2"
    full_latent_path = "data/latent_data/16/RE2500/U_128_latent_full.jld2"
    period_u_path = "data/datasets/RE2500/U_128_period.jld2"
    full_u_path = "data/datasets/RE2500/U_128_full.jld2"
end

function get_NODE_data(period_latent_path, data_path; downsample=-1, clip_bc=true)
    @load period_latent_path z
    @load data_path data
    # preprocess_data!(data; n_samples=downsample, clip_bc=clip_bc, verbose=false)

    z = Float32.(cat(z...; dims=2))
    t = Float32.(data["time"])
    t .-= t[1]

    tspan = (t[1], t[end])
    z0 = z[:, 1]
    @info "Instantiated latent NODE data" size(z) size(t) tspan size(z0)
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

    nn = Chain(
        Dense(latent_dim, dense_mult * latent_dim, activation),
        Dense(dense_mult * latent_dim, latent_dim))
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
# - If setup_method == :lux, `p` is expected to be the ComponentArray returned by Lux.setup and st is used.
function predict(node::NODE, z0; p=nothing, t=nothing)
    tspan = node.tspan
    solver = node.solver
    t = t === nothing ? node.t : t

    p_used = p === nothing ? node.p0 : p
    nnode = NeuralODE(node.dudt, tspan, solver; saveat=t, abstol=node.abstol, reltol=node.reltol)
    sol = nnode(z0, p_used, node.st)

    if isa(sol, Tuple)
        sol = sol[1]
    end

    return sol
end


# Convenience: produce a (latent_dim, n_timepoints) Array prediction like in prelatent_NODE.jl
function predict_array(node::NODE, z0; p=nothing, t=nothing)
    sol = predict(node, z0; p=p, t=t)
    return hcat(sol.u...) |> Array
end

# Mean-squared error loss between data `z` and NODE prediction (z shaped like (latent_dim, n_t))
function mse_loss(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing)
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

function save_node(path::AbstractString, node::NODE, args::NodeArgs)
    args_copy = deepcopy(args)
    node_tspan = node.tspan
    node_t = node.t
    @save path node_p0 = node.p0 node_st = node.st node_args = args_copy node_tspan node_t
    @info "Saved NODE to $path"
end

function load_node(path::AbstractString)
    @load path node_p0 node_st node_args node_tspan node_t
    node = NODE(node_args.latent_dim, node_args.dense_mult; activation=node_args.activation, verbose=false)
    setup_lux!(node, verbose=false)   # create a matching structure
    node.p0 = node_p0
    node.st = node_st
    node.tspan = node_tspan
    node.t = node_t
    @info "NODE sucessfully loaded from $path"
    return node, node_args
end

function plot_node_trajectory(node::NODE, z::AbstractMatrix, z0; p=nothing, t=nothing, n_reconstruct=4)
    idx_samples = round.(Int, range(1, stop=size(z, 1), length=n_reconstruct))
    z_samples = [vec(z[i, :]) for i in idx_samples]  # Vector of 8 one-dimensional arrays, each length 179
    ẑ = predict_array(node, z0; p=p, t=t)
    ẑ_samples = [vec(ẑ[i, :]) for i in idx_samples]  # Vector of 4 one-dimensional arrays, each length 179
    p = plot()
    colors = [:black, :red, :blue, :green, :purple, :orange, :yellow]
    for i in 1:n_reconstruct
        plot!(p, node.t, z_samples[i]; color=colors[i], label="data_$i")
        plot!(p, node.t, ẑ_samples[i]; linestyle=:dash, color=colors[i], label="pred_$i")
    end
    return p
end
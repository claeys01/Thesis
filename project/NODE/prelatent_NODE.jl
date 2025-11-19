using JLD2 
using Random
using Revise
using DiffEqFlux, ComponentArrays, Zygote, OrdinaryDiffEq,
      Printf, Random, MLUtils, OneHotArrays, Lux
using Optimization, OptimizationOptimisers
using Plots

includet("../custom.jl")

includet("NODE_core.jl")
# load hyperparamters
args = NodeArgs()
args.seed > 0 && Random.seed!(args.seed)


# ---------------------------
# Data loading / slicing
# ---------------------------
@load args.period_latent_path z
@load args.period_u_path data
preprocess_data!(data; n_samples=args.downsample, clip_bc=args.clip_bc)

z = Float32.(cat(z...;dims=2))
t = Float32.(data["time"])
t .-= t[1]
# @show size(t_new) size(t)

tspan = (t[1], t[end])
z0 = z[:,1]
# z, t, tspan, z0 = get_NODE_data(args.period_latent_path, args.period_u_path)


idx_samples = round.(Int, range(1, stop=size(z, 1), length=4))
z_samples = [vec(z[i, :]) for i in idx_samples]  # Vector of 8 one-dimensional arrays, each length 179


@show size(z), typeof(z)

# define dudt
dudt = Chain(
    Dense(args.latent_dim, args.dense_mult * args.latent_dim, tanhshrink),
    Dense(args.dense_mult * args.latent_dim, args.latent_dim),
)

# initalize model weights
ps, st = Lux.setup(Xoshiro(0), dudt)


# define latent NODE
nn_ode = NeuralODE(dudt, tspan, Tsit5(); saveat = t)


function predict(p)
    out = nn_ode(z0, p, st)              # may be ODESolution or (ODESolution, newstate)
    ode_sol = isa(out, Tuple) ? out[1] : out
    # build (latent_dim, n_timepoints) array matching `z`
    pred = hcat(ode_sol.u...) |> Array
    return pred
end

function loss_neuralode(p)
    pred = predict(p)
    @assert size(pred) == size(z) "pred $(size(pred)) vs z $(size(z))"
    loss = sum(abs2, z .- pred)
    return loss
end

callback_step = Ref(0)


function callback(state, l; plotting=true)
    callback_step[] += 1

    @info "solver callback call $(callback_step[]): loss = $(l)"
    z_pred = predict(state.u)
    z_pred_samples = [vec(z_pred[i, :]) for i in idx_samples]  # Vector of 4 one-dimensional arrays, each length 179

    if plotting
        p = plot()
        colors = [:black, :red, :blue, :green]
        for i in 1:4
            plot!(p, t, z_samples[i]; color=colors[i], label = "data_$i")
            plot!(p, t, z_pred_samples[i]; linestyle = :dash, color=colors[i], label = "pred_$i")
        end
        display(p)
    end
    return false
end

pinit = ComponentArray(ps)
# callback((; z = pinit), loss_neuralode(pinit))

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, ps) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)


result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 500)

# -------------------------
# Save final plot & params
# -------------------------
using Dates
out_dir = joinpath("data", "NODE_models", Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"))
mkpath(out_dir)

# build final plot with optimized parameters
optimized_params = result_neuralode.u
z_pred_final = predict(optimized_params)
z_pred_samples_final = [vec(z_pred_final[i, :]) for i in idx_samples]

plt_final = plot()
colors = [:black, :red, :blue, :green]
for i in 1:4
    plot!(plt_final, t, z_samples[i]; color = colors[i], label = "data_$i")
    plot!(plt_final, t, z_pred_samples_final[i]; linestyle = :dash, color = colors[i], label = "pred_$i")
end

png_path = joinpath(out_dir, "latent_fit.png")
savefig(plt_final, png_path)

# save optimized parameters
params_path = joinpath(out_dir, "optimized_params.jld2")
@info "Saving plot -> $png_path and params -> $params_path"
@save params_path optimized_params

# Test: load saved params and use them to predict a trajectory
try
    saved = JLD2.load(params_path)
    @info "Loaded keys from $params_path: $(collect(keys(saved)))"
    p_loaded = haskey(saved, "optimized_params") ? saved["optimized_params"] : first(values(saved))

    try
        pred_loaded = predict(p_loaded)
        @info "predict(p_loaded) succeeded, prediction size = $(size(pred_loaded))"
    catch err1
        @warn "predict(p_loaded) failed: $err1 — will try mapping into ComponentArray(ps)"
        p_struct = ComponentArray(ps)
        try
            p_struct .= p_loaded
            pred_mapped = predict(p_struct)
            @info "predict(mapped params) succeeded, prediction size = $(size(pred_mapped))"
        catch err2
            @error "Mapping loaded params into ComponentArray and predicting failed: $err2"
        end
    end
catch err
    @error "Could not load params from $params_path: $err"
end


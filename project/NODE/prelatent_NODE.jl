using JLD2 
using Random
using Revise
using DiffEqFlux, ComponentArrays, Zygote, OrdinaryDiffEq,
      Printf, Random, MLUtils, OneHotArrays, Lux
using Optimization, OptimizationOptimisers
using Plots

includet("Lux_AE.jl")
includet("../utils/AE_normalizer.jl")
includet("../custom.jl")


# load hyperparamters
args = LuxArgs()
args.seed > 0 && Random.seed!(args.seed)

dense_mult = 2
nt = 5          # training trajectory length (steps per loss)

# ---------------------------
# Data loading / slicing
# ---------------------------
@load "/home/matth/Thesis/data/latent_data/128_u_biot_data_arr_force_period.jld2" z
@load args.data_path data
preprocess_data!(data; n_samples=args.downsample, clip_bc=args.clip_bc)

z = Float64.(cat(z...;dims=2))
t = Float64.(data["time"])
tspan = (t[1], t[end])
z_0 = z[:,1]

idx_samples = round.(Int, range(1, stop=size(z, 1), length=4))
z_samples = [vec(z[i, :]) for i in idx_samples]  # Vector of 8 one-dimensional arrays, each length 179




# tsteps = range(tspan[1], tspan[2]; length = length(time))


@show size(z), typeof(z)

# define dudt
dudt = Chain(
    Dense(args.latent_dim, dense_mult * args.latent_dim, tanh),
    Dense(dense_mult * args.latent_dim, args.latent_dim),
)

# initalize model weights
ps, st = Lux.setup(Xoshiro(0), dudt)

# define latent NODE
nn_ode = NeuralODE(dudt, tspan, Tsit5(); saveat = t)


function predict(p)
    out = nn_ode(z_0, p, st)              # may be ODESolution or (ODESolution, newstate)
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


function callback(state, l;)
    # println(l)
    callback_step[] += 1

    @info "solver callback call $(callback_step[]): loss = $(l)"
    # @show typeof(state)
    z_pred = predict(state.u)
    z_pred_samples = [vec(z_pred[i, :]) for i in idx_samples]  # Vector of 4 one-dimensional arrays, each length 179

    p = plot()
    colors = [:black, :red, :blue, :green]
    for i in 1:4
        plot!(p, t, z_samples[i]; color=colors[i])
        plot!(p, t, z_pred_samples[i]; linestyle = :dash, color=colors[i])
    end
    display(p)
    return false
end

pinit = ComponentArray(ps)
# callback((; z = pinit), loss_neuralode(pinit))

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, ps) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)


result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)


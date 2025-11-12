using JLD2 
using Random
using Revise
using DiffEqFlux, ComponentArrays, Zygote, OrdinaryDiffEq,
      Printf, Random, MLUtils, OneHotArrays, Lux
using Optimization, OptimizationOptimisers

includet("Lux_AE.jl")
includet("../utils/AE_normalizer.jl")
includet("../custom.jl")


# load hyperparamters
args = LuxArgs()
args.seed > 0 && Random.seed!(args.seed)

dense_mult = 4
nt = 5          # training trajectory length (steps per loss)

# ---------------------------
# Data loading / slicing
# ---------------------------
@load "/home/matth/Thesis/data/latent_data/128_u_biot_data_arr_force_period.jld2" z
z = cat(z...;dims=2)
z_0 = z[:,1]

@load args.data_path data
preprocess_data!(data; n_samples=args.downsample, clip_bc=args.clip_bc)
time = data["time"]
tspan = (time[1], time[end])

@show size(z)

# define dudt
dudt = Chain(
    Dense(args.latent_dim, dense_mult * args.latent_dim, tanh),
    Dense(dense_mult * args.latent_dim, args.latent_dim),
    x -> 0.1f0 .* tanh.(x)    # scale & bound the final derivative
)

# define latent NODE
nn_ode = NeuralODE(dudt, tspan, Tsit5())

# convert ODE solution to array for decoder
solution_to_array(sol) = sol.u[end]

# define total model
m = Chain(
    nn_ode,
    solution_to_array,
)

# initalize model weights
ps, st = Lux.setup(Xoshiro(0), m);
ps = ComponentArray(ps);
st = st;

function predict(x)
    Array(m(z_0, x, st)[1])
end

function loss_neuralode(p)
    pred = predict(p)
    loss = sum(abs2, z .- pred)
    return loss
end


function callback(state, l)
    println(l)
    return false
end

pinit = ComponentArray(ps)
callback((; z = pinit), loss_neuralode(pinit))

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, ps) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)


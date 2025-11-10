using JLD2 
using Random
using Revise
using DiffEqFlux, ComponentArrays, Zygote, OrdinaryDiffEq,
      Printf, Random, MLUtils, OneHotArrays, Lux
using Optimization, OptimizationOptimisers

includet("Lux_AE.jl")
includet("../custom.jl")


# load hyperparamters
args = LuxArgs()
args.seed > 0 && Random.seed!(args.seed)

dense_mult = 4
nt = 5          # training trajectory length (steps per loss)

# ---------------------------
# Data loading / slicing
# ---------------------------

@load args.data_path RHS_data
downsample_RHS_data!(RHS_data; n_samples=args.downsample, clip_bc=args.clip_bc)

u_vec = RHS_data["u"]
μ₀ = RHS_data["μ₀"]
time = RHS_data["time"]
tspan = (time[1], time[end])
println(tspan)



u = cat(u_vec...; dims = 4)
μ₀ = cat(μ₀...; dims=4)

u_in = cat(u, μ₀; dims=3)
u_0 = u_in[:,:,:,1:1]


@assert ndims(u_in) == 4 "RHS_data[\"u\"] must be (H,W,C,T). Got size $(size(u_in))"
@show size(u_in)
@show size(u_0)


# initalize encoder & decoder
encoder = Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride)
decoder = Decoder(args.output_dim, args.latent_dim; C_next=args.C_conv)

# define dudt
dudt = Chain(
    Dense(args.latent_dim, dense_mult * args.latent_dim, relu),
    Dense(dense_mult * args.latent_dim, args.latent_dim)
)

# define latent NODE
nn_ode = NeuralODE(dudt, tspan, Tsit5())

# convert ODE solution to array for decoder
solution_to_array(sol) = sol.u[end]

# define total model
m = Chain(
    encoder.layers,
    nn_ode,
    solution_to_array,
    decoder.layers
)

# initalize model weights
ps, st = Lux.setup(Xoshiro(0), m);
ps = ComponentArray(ps);
st = st;

# random single-sample input matching args.input_dim
x = rand(Float32, args.input_dim[1], args.input_dim[2], args.input_dim[3], 178)

# forward pass through your Lux model `m` using `ps, st` from Lux.setup
y, st2 = m(x, ps, st)

# inspect result
@show size(x)
@show size(y)
@show typeof(y)

function predict(x)
    Array(m(u_0, x, st)[1])
end

function loss_neuralode(p)
    pred = predict(p)
    loss = sum(abs2, u .- pred)
    return loss
end


function callback(state, l)
    println(l)
    return false
end

pinit = ComponentArray(ps)
callback((; u = pinit), loss_neuralode(pinit))

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, ps) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

# result_neuralode = Optimization.solve(
#     optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)

# quick checks for NaNs/Infs, shape and a direct dudt test
latent, st_enc = encoder(u_0, ps, st)   # get the encoded initial latent
println("tspan=", tspan, " any NaN in tspan? ", any(isnan, Tuple(tspan)))
println("size(u_0)=", size(u_0))
println("size(latent)=", size(latent))
println("any NaN/Inf in latent? ", any(isnan, Array(latent)), any(isinf, Array(latent)))
println("any NaN/Inf in params? ", any(x -> any(isnan, Array(x)) || any(isinf, Array(x)), values(ps)))

# test dudt directly on that latent (convert to plain Array/vector if needed)
z = Array(latent)                     # ensure plain Array/Float32
println("dudt(z) -> size: ", size(dudt(z)))
println("any NaN/Inf in dudt(z)? ", any(isnan, Array(dudt(z))), any(isinf, Array(dudt(z))))

# try random input to see if dudt can produce finite outputs
r = randn(Float32, length(z))
println("dudt(randn) any NaN? ", any(isnan, Array(dudt(r))), any(isinf, Array(dudt(r))))

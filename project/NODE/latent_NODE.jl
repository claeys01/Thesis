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

@load args.data_path data
preprocess_data!(data; n_samples=args.downsample, clip_bc=args.clip_bc)

u, normalizer = normalize_batch(data["u"]; normalizer=nothing)
μ₀ = data["μ₀"]
time = data["time"]
tspan = (time[1], time[end])
println(tspan)


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

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)

# rng = Xoshiro(0)
# ps_enc, st_enc = Lux.setup(rng, encoder.layers)

# # debug: check NaN/Inf and scales before calling NeuralODE
# z, _ = encoder.layers(u_0, ps_enc, st_enc)       # latent initial condition
# @show any(isnan, z), any(isinf, z), maximum(abs, z)

# p_node, st_node = Lux.setup(rng, dudt)
# dz = dudt(z, p_node, st_node)[1]                  # single-step dudt output
# @show any(isnan, dz), any(isinf, dz), maximum(abs, dz)

# # try evaluate nn_ode once (loose tolerances) to see if it explodes
# nn_ode_test = NeuralODE(dudt, tspan, Tsit5(); reltol=1e-6, abstol=1e-6)
# try
#     sol = nn_ode_test(z[:, :, :, 1:1], p_node, st_node)   # or pass shaped latent vec expected by your chain
#     @show sol[end]
# catch e
#     @error "NeuralODE forward failed: $e"
# end

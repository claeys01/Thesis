using JLD2 


includet("../AE/AE_core.jl")


@load "data/latent_data/128_RHS_biot_data_arr_force_period.jld2" z
@load "data/datasets/128_RHS_biot_data_arr_force_period.jld2" RHS_data




# load hyperparamters
args = Args(; kws...)
args.seed > 0 && Random.seed!(args.seed)

dense_mult = 4


# initalize encoder & decoder
encoder = Flux.f32(Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride))
decoder = Flux.f32(Decoder(args.output_dim, args.latent_dim; C_next=args.C_conv)) 

# define dudt
dudt = Chain(
    Dense(args.latent_dim, dense_mult * args.latent_dim, relu),
    Dense(dense_mult * args.latent_dim, args.latent_dim)
)

# define latent NODE
nn_ode = NeuralODE(dudt, (0.0f0, 1.0f0), Tsit5(); save_everystep = false,
    sensealg = BacksolveAdjoint(; autojacvec = ZygoteVJP()),
    reltol = 1e-5, abstol = 1e-6, save_start = false
)

# define total model
m = Chain(
    encoder,
    nn_ode,
    decoder
)

# initalize model weights
ps, st = Lux.setup(Xoshiro(0), m);
ps = ComponentArray(ps);
st = st;




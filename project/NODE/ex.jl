includet("NODE_core.jl")
includet("../AE/Lux_AE.jl")

args = NodeArgs()
@show typeof(args)
z, t, tspan, z0 = get_NODE_data(args.train_latent_path; downsample=args.downsample)


node = NODE(args.latent_dim, args.dense_mult; 
            tspan=tspan, t=t, activation=args.activation, 
            solver=args.solver, abstol=args.abstol, reltol=args.reltol)
setup_lux!(node)

@show typeof(node)

AEargs = LuxArgs()

enc = Encoder(AEargs, verbose=true)
dec = Decoder(AEargs, verbose=true)
ae = AE(enc, dec)
rng = Xoshiro(args.seed)
ps, st = Lux.setup(rng, ae)

# @show  typeof(enc), typeof(dec)
@show typeof(ae)


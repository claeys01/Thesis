using Thesis
using Statistics
using Plots
using CUDA
using Lux
using Random

simdata_path = "data/inline_runs/hpc_inline/U_inline.jld2"

# simdata = load_simdata(simdata_path)

# force_plot = plot(simdata.time, last.(simdata.force))

# display(force_plot)

# ae_bundle, args = load_trained_AE(checkpoint)
# get_latent_vectors
# args=LuxArgs(;)
# enc = Encoder(args)
# dec=Decoder(args)
# ae=AE(enc, dec)

 data, loaders, normalizer = Thesis.get_data(
            args.batch_size,
            args.full_data_path;
            n_training = args.train_downsample,
            n_test = args.test_downsample,
            split = args.split,
            t_training = args.t_training,
            simdata_ram=args.simdata_ram,
            plotpath="figs/train_force_split.pdf",
            showplot=true
        )
GC.gc()
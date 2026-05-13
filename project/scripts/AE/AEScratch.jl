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
args=LuxArgs(;)
enc = Encoder(args)
dec=Decoder(args)
ae=AE(enc, dec)
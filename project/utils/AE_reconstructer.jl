using JLD2
using Random
using Plots
using Flux
using MLUtils: DataLoader

includet("../AE/conv_AE_zoo_cpu.jl")  # to get Encoder, Decoder, get_data, reconstruct, Args
includet("../custom.jl")



function visualize_reconstructions(checkpoint_path; n::Int=5, seed::Int=42, device=Flux.get_device("CPU"), savepath=nothing)
    # load checkpoint (expects keys "encoder", "decoder", "args")
    checkpoint = JLD2.load(checkpoint_path)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    args = Args(; checkpoint["args"]...)
    encoder = Encoder(args.input_dim, args.latent_dim)
    decoder = Decoder(args.input_dim, args.latent_dim)
    Flux.loadmodel!(encoder, encoder_state)
    Flux.loadmodel!(decoder, decoder_state)

    args.seed > 0 && Random.seed!(args.seed)

    @load args.data_path RHS_data
    snapshots, ids = get_random_snapshots(args.data_path, downsample=args.downsample)

    



end

visualize_reconstructions("/home/matth/Thesis/data/saved_models/first/checkpoint.jld2")
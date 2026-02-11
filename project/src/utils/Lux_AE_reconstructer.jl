using Thesis
using Random

function visualize_reconstructions(checkpoint_path::Union{String,Nothing}=nothing;
                                   device=cpu_device())
    """
    Visualize reconstructions.

    Usage:
      - provide checkpoint_path (String) -> load encoder/decoder and args from checkpoint
    """
    # # load from checkpoint
    # checkpoint = load(checkpoint_path)

    # ps = checkpoint["ps"]
    # st = checkpoint["st"]
    # normalizer = checkpoint["normalizer"]
    # # reconstruct args: try LuxArgs then fallback to Args
    # args_dict = checkpoint["args"]
    # args = LuxArgs(; args_dict...)

    # # determine runtime device (unless explicitly passed)
    # if device === cpu_device() && hasproperty(args, :use_gpu)
    #     device = args.use_gpu ? gpu_device() : cpu_device()
    # end
    # cpu = cpu_device()
    
    # # move params/states to device
    # ps = device(ps)
    # st = device(st)
    # st_test = LuxCore.testmode(st)

    # # Move normalizer to device if normalizing
    # if args.normalize && normalizer !== nothing
    #     normalizer = Normalizer(
    #         device(Float32.(normalizer.μ)),
    #         device(Float32.(normalizer.σ)),
    #         normalizer.method
    #     )
    # end

    # enc = Encoder(args, verbose=false)
    # dec = Decoder(args, verbose=false)

    # ae = AE(enc, dec)
    enc, dec, ae, ps, st, args = load_trained_AE(checkpoint_path; return_params=true) .|> cpu_device()
    normalizer = load_normalizer(checkpoint_path)
    # optional seeding
    args.seed > 0 && Random.seed!(args.seed)

    simdata = load_simdata(args.full_data_path)
    preprocess_data!(simdata; verbose=false)

    span = length(simdata.time)

    if span < args.n_reconstruct
        error("amount of data must be ≥ $(args.n_reconstruct)")
    end

    ids = randperm(span)[1:args.n_reconstruct]

    @info "Selected $(args.n_reconstruct) snapshots with indices $ids for reconstruction"
    plots = []
    dirs = ["x" , "y"]
    for (s, id) in enumerate(ids)
        μ₀ = simdata.μ₀[:, :, :, id:id]
        x = simdata.u[:, :, :, id:id]
        if args.normalize
            # Move data to device BEFORE normalization
            x_dev = device(Float32.(x))
            μ₀_dev = device(Float32.(μ₀))
            
            x_target_dev, _ = normalize_batch(x_dev; normalizer=normalizer)
            x_in_dev = cat(x_target_dev, μ₀_dev; dims=3)
            
            x̂_norm, _ = ae(x_in_dev, ps, st_test)
            
            # Denormalize on device, then move to CPU
            x̂_dev = denormalize_batch(x̂_norm, normalizer)
            x_target_cpu = denormalize_batch(x_target_dev, normalizer)
            
            # Move to CPU for plotting
            x̂ = cpu(Array(x̂_dev))
            μ₀ = cpu(Array(μ₀_dev))
            x_target = cpu(Array(x_target_cpu))
        else
            x_in = cat(x, μ₀; dims=3)
            x_in_dev = device(Float32.(x_in))
            x̂_dev, _ = ae(x_in_dev, ps, st_test)
            x̂ = cpu(Array(x̂_dev))
            x_target = x
        end

        for ch in 1:2
            mat_in = x_target[:, :, ch, 1]
            mat_out = x̂[:, :, ch, 1] .* μ₀[:, :, ch, 1]


            μ = mean(mat_in)
            σ = std(mat_in)
            clim = (μ - σ, μ + σ)

            img_in = flood(mat_in;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title="snapshot $(ids[s]): Input $(dirs[ch])",
                titlefontsize=8)

            img_out = flood(mat_out;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title="snapshot $(ids[s]): Reconstructed $(dirs[ch])",
                titlefontsize=8)

            push!(plots, img_in)
            push!(plots, img_out)
        end
    end
    
    p = plot(plots...;
             layout=(args.n_reconstruct*2, 2),
             link=:none, legend=false,
             size=(500, 750),
             dpi=200, grid=false)

    return p
end

if isinteractive()
    p = visualize_reconstructions("data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb11-1156__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1/checkpoint.jld2")
end

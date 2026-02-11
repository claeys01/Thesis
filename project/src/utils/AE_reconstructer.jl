function visualize_reconstructions(checkpoint_path::Union{String,Nothing}=nothing;
                                   device=cpu_device())
    """
    Visualize reconstructions.

    Usage:
      - provide checkpoint_path (String) -> load encoder/decoder and args from checkpoint
    """

    cpu = cpu_device()
    _, _, ae, ps, st, args = load_trained_AE(checkpoint_path; return_params=true)
    ps = cpu(ps)
    st = cpu(st)

    normalizer = load_normalizer(checkpoint_path)
    

    args.seed > 0 && Random.seed!(args.seed)

    simdata = load_simdata(args.full_data_path)
    preprocess_data!(simdata; verbose=false)

    span = length(simdata.time)

    if span < args.n_reconstruct
        error("amount of data must be ≥ $(args.n_reconstruct)")
    end

    ids = randperm(span)[1:args.n_reconstruct]
    t = round.(simdata.time[ids], digits=2)

    @info "Selected $(args.n_reconstruct) snapshots with indices $ids for reconstruction"
    plots = []
    dirs = ["x" , "y"]
    for (s, id) in enumerate(ids)
        μ₀ = simdata.μ₀[:, :, :, id:id]
        x = simdata.u[:, :, :, id:id]
        if args.normalize
            # Move data to device BEFORE normalization
            x_dev = cpu(Float32.(x))
            μ₀_dev = cpu(Float32.(μ₀))
            
            x_target_dev, _ = normalize_batch(x_dev; normalizer=normalizer)
            x_in_dev = cat(x_target_dev, μ₀_dev; dims=3)
            
            x̂_norm, _ = ae(x_in_dev, ps, st)
            
            mae = mean(abs, (x̂_norm .* μ₀_dev .- x̂_norm))
            corr = cor(vec(x̂_norm .* μ₀_dev), vec(x̂_norm))


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
            x̂_dev, _ = ae(x_in_dev, ps, st)
            x̂ = cpu(Array(x̂_dev))
            x_target = x
        end

        for ch in 1:2
            mat_in = x_target[:, :, ch, 1]
            mat_out = x̂[:, :, ch, 1] .* μ₀[:, :, ch, 1]

            # mae = mean(abs.(mat_in .- mat_out))
            # corr = cor(vec(mat_in), vec(mat_out))

            μ = mean(mat_in)
            σ = std(mat_in)
            clim = (μ - σ, μ + σ)

            img_in = flood(mat_in;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title=" time: $(t[s]): Input $(dirs[ch])",
                titlefontsize=8)

            img_out = flood(mat_out;
                border=:none, colorbar=false, framestyle=:none,
                axis=nothing, ticks=false, clims=clim,
                aspect_ratio=:equal,
                title="Recon $(dirs[ch]) | MAE=$(round(mae; sigdigits=3)) | r=$(round(corr; digits=3))",
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


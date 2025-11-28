using JLD2
using Lux


includet("../AE/Lux_AE.jl")
includet("../custom.jl")
includet("../utils/AE_normalizer.jl")


function get_latent_data(checkpoint_path::String; save_path::Union{String,Tuple{String,String},Nothing}=nothing)
    enc, dec, ae, ps, st = load_trained_AE(checkpoint_path; return_params=true)
    
    checkpoint = JLD2.load(checkpoint_path)
    args_dict = checkpoint["args"]
    args = LuxArgs(; args_dict...)
    normalizer = checkpoint["normalizer"]


    period_data = load(args.data_path, "data")
    full_data = load(args.full_data_path, "data")

    preprocess_data!(period_data; clip_bc=args.clip_bc, verbose=true)
    preprocess_data!(full_data; clip_bc=args.clip_bc, verbose=true)


    # helper that computes latent snaps for one dataset dict
    function compute_latents(data::Dict)
        # ensure μ₀ is a 4-D array too if present
        if haskey(data, "μ₀")
            data["μ₀"] = isa(data["μ₀"], AbstractArray) && ndims(data["μ₀"]) == 4 ?
                             Float32.(data["μ₀"]) : Float32.(cat(data["μ₀"]...; dims=4))
        end

        nsnaps = size(data["u"], 4)
        @info "Found $nsnaps snapshots"

        latent_snaps = Vector{Vector{Float32}}()
        for i in 1:nsnaps
            x = data["u"][:, :, :, i]         # H×W×C
            μ₀ = haskey(data, "μ₀") ? data["μ₀"][:, :, :, i] : zeros(Float32, size(x,1), size(x,2), 1)

            # make a (H,W,C,1) input batch
            if args.normalize
                x_target_norm, _ = normalize_batch(reshape(x, size(x,1), size(x,2), size(x,3), 1), normalizer=normalizer)
                x_in = cat(x_target_norm, reshape(μ₀, size(μ₀,1), size(μ₀,2), size(μ₀,3), 1); dims=3)
            else
                x_in = cat(reshape(x, size(x,1), size(x,2), size(x,3), 1),
                           reshape(μ₀, size(μ₀,1), size(μ₀,2), size(μ₀,3), 1); dims=3)
            end

            # pass a 4-D array (H,W,C,N) to the encoder — do NOT wrap in [ ... ]
            z, _ = enc(x_in, ps.encoder, st.encoder)                # z has shape (latent_dim, 1) or similar
            push!(latent_snaps, vec(Array(z)))   # store as 1-D Float32 vector
        end

        return latent_snaps
    end

    @info "Computing latents for period_data"
    period_latent = compute_latents(period_data)

    @info "Computing latents for full_data"
    full_latent = compute_latents(full_data)

    # handle saving: save_path can be nothing, a single string (creates two files with suffixes),
    # or a tuple of two filenames (period_path, full_path)
    if save_path !== nothing
        if isa(save_path, String)
            root, ext = splitext(save_path)
            period_path = string(root, "_period", ext)
            full_path   = string(root, "_full",   ext)
        elseif isa(save_path, Tuple) && length(save_path) == 2
            period_path, full_path = save_path
        else
            error("save_path must be nothing, a String, or a Tuple{String,String}")
        end

        @info "Saving period latents to $period_path"
        @save period_path z = period_latent

        @info "Saving full latents to $full_path"
        @save full_path z = full_latent
    end

    return period_latent, full_latent
end

checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/u_100period_100e_4096n_256h_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
save_path = "data/latent_data/16/RE2500/U_128_latent.jld2"
kkr_period, kkr_full = get_latent_data(checkpoint; save_path=save_path)
nothing

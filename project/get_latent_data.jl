using JLD2
using Flux


includet("AE/AE_core.jl")
includet("custom.jl")
includet("utils/AE_normalizer.jl")


function get_latent_data(checkpoint_path::String; save_path::Union{String,Nothing}=nothing)
    checkpoint = JLD2.load(checkpoint_path)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    normalizer = checkpoint["normalizer"]
    args = Args(; checkpoint["args"]...)
    enc = Encoder(args.input_dim, args.latent_dim; C_next=args.C_conv, padding=args.padding, stride=args.stride, verbose=false)
    dec = Decoder(args.output_dim, args.latent_dim; C_next=args.C_conv, verbose=false)
    Flux.loadmodel!(enc, encoder_state)
    Flux.loadmodel!(dec, decoder_state)

    @load args.data_path RHS_data
    downsample_RHS_data!(RHS_data; clip_bc=args.clip_bc, verbose=true)

    # ensure RHS is a 4-D array (H,W,C,N)
    get_4d!(RHS_data)
    # ensure μ₀ is a 4-D array too if present
    if haskey(RHS_data, "μ₀")
        RHS_data["μ₀"] = isa(RHS_data["μ₀"], AbstractArray) && ndims(RHS_data["μ₀"]) == 4 ?
                         Float32.(RHS_data["μ₀"]) : Float32.(cat(RHS_data["μ₀"]...; dims=4))
    end

    nsnaps = size(RHS_data["RHS"], 4)
    @info "Found $nsnaps snapshots"

    latent_snaps = Vector{Vector{Float32}}()
    for i in 1:nsnaps
        x = RHS_data["RHS"][:, :, :, i]         # H×W×C
        μ₀ = haskey(RHS_data, "μ₀") ? RHS_data["μ₀"][:, :, :, i] : zeros(Float32, size(x,1), size(x,2), 1)

        # make a (H,W,C,1) input batch
        if args.normalize
            x_target_norm, _ = normalize_batch(reshape(x, size(x,1), size(x,2), size(x,3), 1), normalizer=normalizer)
            x_in = cat(x_target_norm, reshape(μ₀, size(μ₀,1), size(μ₀,2), size(μ₀,3), 1); dims=3)
        else
            x_in = cat(reshape(x, size(x,1), size(x,2), size(x,3), 1),
                       reshape(μ₀, size(μ₀,1), size(μ₀,2), size(μ₀,3), 1); dims=3)
        end

        # pass a 4-D array (H,W,C,N) to the encoder — do NOT wrap in [ ... ]
        z = enc(x_in)                # z has shape (latent_dim, 1) or similar
        push!(latent_snaps, vec(Array(z)))   # store as 1-D Float32 vector
    end

    if save_path !== nothing
        @save save_path z = latent_snaps
    end

    return latent_snaps
end

checkpoint = "data/saved_models/100period_100e_4096n_64l_norm_pooling_ups_mu_L2/checkpoint.jld2"
save_path = "data/latent_data/128_RHS_biot_data_arr_force_period.jld2"
kkr = get_latent_data(checkpoint; save_path=save_path)
nothing

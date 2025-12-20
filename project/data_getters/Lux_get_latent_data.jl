using JLD2
using Lux
using MLUtils



includet("../AE/Lux_AE.jl")
includet("../custom.jl")
includet("../utils/AE_normalizer.jl")
includet("../utils/SimDataTypes.jl")
using .SimDataTypes


function get_latent_data(checkpoint_path::String; save_path::Union{String,Tuple{String,String},Nothing}=nothing, batch_size=1024)
    enc, _, _, ps, st = load_trained_AE(checkpoint_path; return_params=true)
    st = LuxCore.testmode(st)

    checkpoint = JLD2.load(checkpoint_path)
    args_dict = checkpoint["args"]
    args = LuxArgs(; args_dict...)
    normalizer = checkpoint["normalizer"]

    simdata = load_simdata(args.full_data_path)
    N =size(simdata.u, 4)

    preprocess_data!(simdata; verbose=true)

    train_idx = findall(t -> t < args.t_training, simdata.time)
    test_idx = collect(last(train_idx)+1 : N)

    TrainData = EpochData(get_data_in(simdata.u, simdata.μ₀; idx=train_idx)...)
    t_train = simdata.time[train_idx]
    train_loader = DataLoader(train_idx, batchsize=batch_size, shuffle=false)

    TestData = EpochData(get_data_in(simdata.u, simdata.μ₀; idx=test_idx)...)
    t_test = simdata.time[test_idx]
    n_test = size(t_test, 1)
    test_loader = DataLoader(collect(1 : n_test); batchsize=batch_size, shuffle=false)


    simdata = nothing

    # helper that computes latent snaps for one dataset dict
    
    function compute_latents(data::EpochData, enc::Encoder, ps, st, normalizer, idx_loader)
        nsnaps = size(data.Xin, 4)
        @info "  Found $nsnaps snapshots"
        
        latents = Matrix{Float32}(undef, args.latent_dim, nsnaps)

        for idx in idx_loader
            Xin, _, _ = build_batch(data, idx; normalizer)
            z, _ = enc(Xin, ps.encoder, st.encoder)
            latents[:, idx] = z
           
        end 
        return latents
    end

    @info "Computing latents for training data"
    # jemoeder = compute_latents(TrainData, enc, ps, st, normalizer, train_loader)
    # @show size(jemoeder), typeof(jemoeder)
    train_latent = LatentData(
        compute_latents(TrainData, enc, ps, st, normalizer, train_loader), 
        t_train
    )
    TrainData = nothing
    GC.gc()

    @info "Computing latents for test data"
    test_latent = LatentData(
        compute_latents(TestData, enc, ps, st, normalizer, test_loader), 
        t_test
    )
    TestData = nothing
    GC.gc()


    # handle saving: save_path can be nothing, a single string (creates two files with suffixes),
    # or a tuple of two filenames (period_path, full_path)
    if save_path !== nothing
        if isa(save_path, String)
            root, ext = splitext(save_path)
            train_path = string(root, "_train", ext)
            test_path   = string(root, "_test",   ext)
        elseif isa(save_path, Tuple) && length(save_path) == 2
            train_path, test_path = save_path
        elses
            error("save_path must be nothing, a String, or a Tuple{String,String}")
        end

        @info "Saving train latents to $train_path"
        @save train_path train_latent

        @info "Saving test latents to $test_path"
        @save test_path test_latent        
    end

    return train_latent, test_latent
end

checkpoint = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/2e8_u_200e_4096n_16l_norm_pooling_ups_mu_L1/checkpoint.jld2"
save_path = "data/latent_data/16/RE2500/2e8/U_128_latent.jld2"
train_latent, test_latent = get_latent_data(checkpoint; save_path=save_path)
nothing

using Thesis
using Thesis: get_trainval_idx, build_batch, get_data_in
using JLD2

"""
    get_latent_vectors(checkpoint_path::String, args::LuxArgs; device=cpu_device())

Load simulation data and encode it into latent vectors using a trained autoencoder.

# Arguments
- `checkpoint_path`: Path to the trained AE checkpoint

# Returns
- `z`: Array of latent vectors (latent_dim × n_samples)
- `t`: Corresponding time points
"""

function get_latent_vectors(ae::AE, ps, st, normalizer, ae_args::LuxArgs; device=cpu_device())
    # Load and preprocess simulation data
    simdata = load_simdata(ae_args.full_data_path)
    preprocess_data!(simdata; verbose=true)
    
    # Get training indices (downsampled)
    train_idx = get_trainval_idx(simdata, ae_args.t_training, ae_args.train_downsample)
    
    # Concatenate velocity with mask for encoder input
    x_in, _, _ = build_batch(EpochData(get_data_in(simdata.u, simdata.μ₀; idx=train_idx)...), 1:ae_args.train_downsample ; normalizer=normalizer)
    
    # Move to device
    x_in = device(x_in)
    
    # Encode to latent space
    z, _ = ae.encoder(x_in, ps.encoder, st.encoder)
    
    # Move back to CPU and convert to Array
    z = Array(cpu_device()(z))
    
    t = simdata.time[train_idx]
    tspan, z0 = (t[1], t[end]), z[:, 1]
    @info "Generated latent vectors" size(z) n_samples=length(train_idx) time_range=tspan

    return z, t, tspan, z0
end



function get_latent_vectors(checkpoint_path::String; device=cpu_device())
    # Load trained autoencoder
    _, _, ae, ps, st, ae_args = load_trained_AE(checkpoint_path; device=device, return_params=true)
    normalizer = load_normalizer(checkpoint_path)
    get_latent_vectors(ae, ps, st, normalizer, ae_args; device=device)
end

AE_path = "data/saved_models/u/Lux/256h_16l/RE2500/2e8/Feb12-1530__E1000_HW256x256_C4to2_nc6_nd2_z16_C8_lr0p001_wd0p0009_bs16_NY_LL1_Tl0p0471/checkpoint.jld2"


z, t, tspan, z0 = get_latent_vectors(AE_path)
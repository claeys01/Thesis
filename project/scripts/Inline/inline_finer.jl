using Thesis
using WaterLily
using WaterLily: MeanFlow
using Statistics
using Dates
using JLD2
using Plots


root_path = ""
if is_hpc()
    root_path = "/scratch/mfbclaeys"
    # Log job info
    @info "Starting HPC AE+NODE retrain pipeline"
    @info "  SLURM_JOB_ID: $(get(ENV, "SLURM_JOB_ID", "N/A"))"
    @info "  SLURM_NTASKS: $(get(ENV, "SLURM_NTASKS", "N/A"))"
    @info "  SLURM_CPUS_PER_TASK: $(get(ENV, "SLURM_CPUS_PER_TASK", "N/A"))"
    @info "  Hostname: $(gethostname())"
    @info "  Julia threads: $(Threads.nthreads())"
end

params = InlineParams(
    t_run = 10,
    t_train = 7.5,
    t_accel_end = 50,
    save_interval = 0.25,
    sample_interval = 0.01,
    ae_epochs = 1000,
    ae_retrain_epochs = 300,
    node_iters = 250,
    node_retrain_iters = 100,
    n_switch = 150,
    max_retrain_flags = 3,
)

savedir = joinpath(root_path, "data", "inline_runs", "finer_" * Dates.format(now(), "yyyy-mm-dd_HH-MM"))
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline_finer.jld2")

# sim = circle_shedding_biot(n=2^8, m=2^8; mem=Array, perturb=false)
# u₀ = load_u0("data/datasets/RE2500/2e8/U_128_full_u0.jld2")

sim = circle_shedding_biot(n=2^9, m=2^9; mem=Array, perturb=false)
u₀ = load_u0(joinpath(root_path, "data/initial_fields/u0_biot_n512_t50.jld2"))

hs = HybridState(sim, nothing, params, savedir, nothing, nothing)

simdata = run_warmup!(hs, params.t_run; u₀=u₀, save_path=simdata_path)

display(Thesis.train_force_plot(simdata))

# ================================ Step 1: Train Autoencoder ================================
@info "── Step 1/4: Training Autoencoder ──"
ae_start = time()

div = 100.0
curl = 100.0

@info "AE hyperparameters" epochs=params.ae_epochs λdiv=div λcurl=curl

ae_args = LuxArgs(
        epochs=params.ae_epochs, 
        save_path=savedir,
        λdiv=Float64(div), 
        λcurl=Float64(curl),
        train_downsample=500,
        input_dim = (2^9, 2^9, 4),   # flow field size with μ₀ concatenated
        output_dim = (2^9, 2^9, 2),  # size of reconstructed RHS field
        n_conv = 8,
        batch_size=4,
        n_dense= 2,
        t_training=params.t_train,
        full_data_path=simdata_path, 
        simdata_ram=simdata,
    )

ae_bundle, AE_path = train_AE(ae_args; return_path=true)
ae_elapsed = round((time() - ae_start) / 60; digits=1)
@info "AE initial training complete" elapsed_min=ae_elapsed checkpoint=AE_path

# ae_args.simdata_ram = nothing   # release the simdata ref
normalizer = load_normalizer(AE_path)

ae_bundle = cpu_device()(ae_bundle)

# ================================ Step 2: Train NODE ================================
@info "── Step 2/4: Training Neural ODE ──"
node_args = NodeArgs(
        save_path=savedir,
        maxiters = params.node_iters,
        extrapolate = false,
        use_gpu = false,
        latent_dim = ae_args.latent_dim,
    )
node_start = time()
node, node_path = train_NODE(
    node_args;
    ae_bundle = ae_bundle,
    normalizer = normalizer,
    ae_args = ae_args,
)
node_elapsed = round((time() - node_start) / 60; digits=1)
@info "NODE training complete" elapsed_min=node_elapsed node_path=node_path

# ae_bundle = cpu_device()(ae_bundle)

aenode = AENODE(ae_bundle, node, ae_args, node_args, normalizer; verbose=true)

# hs = HybridState(sim, aenode, params, savedir, AE_path_tl1, node_path)
hs.aenode = aenode
hs.AE_path = AE_path
hs.node_path = node_path

run_hybrid!(hs)
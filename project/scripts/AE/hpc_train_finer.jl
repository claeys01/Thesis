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
    @info "Starting HPC AE pipeline"
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
    node_iters = 1,
    node_retrain_iters = 100,
    n_switch = 150,
    max_retrain_flags = 3,
)

savedir = joinpath(root_path, "data", "Lux_models")
mkpath(savedir)
simdata_path = joinpath(savedir, "U_inline_finer.jld2")

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
        n_dense= 2,
        t_training=params.t_train,
        full_data_path=simdata_path, 
        simdata_ram=simdata,
    )

ae_bundle, AE_path = train_AE(ae_args; return_path=true)
ae_elapsed = round((time() - ae_start) / 60; digits=1)
@info "AE initial training complete" elapsed_min=ae_elapsed checkpoint=AE_path
module Thesis

# ═══════════════════════════════════════════════════════════════════════════════
# Core Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
using Reexport

# Numerics & Arrays
using Statistics
using Random
using LinearAlgebra

# Data handling
using JLD2
using DrWatson: struct2dict

# Progress & Timing
using ProgressMeter
@reexport using TimerOutputs  # <-- Changed: now @timeit is available to users
const to = TimerOutput()
using Dates

# ═══════════════════════════════════════════════════════════════════════════════
# Visualization - Always loaded (needed by custom.jl and other modules)
# ═══════════════════════════════════════════════════════════════════════════════
using Plots
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Machine Learning Stack
# ═══════════════════════════════════════════════════════════════════════════════
@reexport using Lux
using LuxCore
using NNlib
using MLUtils
using MLUtils: DataLoader
using Optimisers
using NearestNeighbors


# Automatic differentiation
using Zygote
using Enzyme

# Neural ODEs
using DiffEqFlux
using OrdinaryDiffEq
using ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationPolyalgorithms

# ═══════════════════════════════════════════════════════════════════════════════
# GPU/Accelerator Support (conditional loading)
# ═══════════════════════════════════════════════════════════════════════════════
using Reactant

const USE_CUDA = Ref(false)

function __init__()
    if get(ENV, "THESIS_USE_CUDA", "false") == "true"
        try
            @eval begin
                using CUDA
                using cuDNN
            end
            
            # Use invokelatest to call methods from newly loaded packages
            cuda_functional = Base.invokelatest(CUDA.functional)
            
            if cuda_functional
                USE_CUDA[] = true
                
                # Get device info using invokelatest
                dev = Base.invokelatest(CUDA.device)
                cap = Base.invokelatest(CUDA.capability, dev)
                @info "CUDA loaded successfully" device=dev capability=cap
                
                # Verify cuDNN is available
                has_cudnn = Base.invokelatest(cuDNN.has_cudnn)
                if has_cudnn
                    cudnn_ver = Base.invokelatest(cuDNN.version)
                    @info "cuDNN loaded successfully" version=cudnn_ver
                else
                    @warn "cuDNN loaded but not functional"
                end
            else
                @warn "CUDA loaded but not functional (no GPU detected)"
            end
        catch e
            @warn "CUDA/cuDNN requested but failed to load" exception=(e, catch_backtrace())
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Dependencies
# ═══════════════════════════════════════════════════════════════════════════════
using WaterLily
import WaterLily: ∂, @loop, @inside, inside_u, S, conv_diff!, δ, CI, inside, flood

using BiotSavartBCs
using BiotSavartBCs: BiotSimulation

# ═══════════════════════════════════════════════════════════════════════════════
# Project Includes - Order matters for dependencies!
# ═══════════════════════════════════════════════════════════════════════════════

# Utilities first (no internal dependencies)
include("utils/SimDataTypes.jl")
using .SimDataTypes: SimData, EpochData, LatentData

include("utils/AE_normalizer.jl")
# Custom functions (depends on WaterLily)
include("utils/custom.jl")

# Plotting utilities
include("utils/plotting_funcs.jl")

# Simulations
include("simulations/vortex_shedding_biot_savart.jl")
include("simulations/vortex_shedding.jl")

# Core model definitions
include("AE/AE.jl")
include("AE/AE_train.jl")

# OOD detection
include("OOD/KNN.jl")

# NODE components
include("NODE/NODE_core.jl")
include("NODE/NODE_train.jl")
include("NODE/NODE_PostTrain.jl")

# Combined AE+NODE
include("AENODE.jl")

# Reconstruction utilities
include("utils/AE_reconstructer.jl")
include("utils/AE_loss_plot.jl")

# Acceleration benchmarking
include("hybrid/acceleration.jl")
include("hybrid/HybdridState.jl")


# Data getters
# include("data_getters/Lux_get_latent_data.jl")

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Types
# ═══════════════════════════════════════════════════════════════════════════════
export SimData, EpochData, LatentData
export LuxArgs, NodeArgs
export Encoder, Decoder, AE
export NODE, AENODE
export Normalizer
export BiotSimulation
export InlineParams, HybridState

# OOD Detection Types
export KNNOOD

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - AE Functions
# ═══════════════════════════════════════════════════════════════════════════════
export train_AE
export load_trained_AE
export visualize_reconstructions
export total_loss, recon_loss, div_loss_L2, masked_loss
# export divergence_field, vorticity_field
# export wake_mask, latent_smoothness, latent_pca_energy
# export pass_fail, compute_baseline_divergence, evaluate_checkpoint, build_eval_row
# export HPARAM_COLS, METRIC_COLS, STATUS_COLS
export load_simdata
export save_u0, load_u0

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - NODE Functions
# ═══════════════════════════════════════════════════════════════════════════════
export load_node, train_NODE
export predict_array, predict
export predict_n, predict_n!
export predict_flex, predict_flex!
export NODE, setup_lux!
export L2_loss, loss_multiple_shoot, loss_multiple_shoot_multi
export plot_node_trajectory, plot_multiple_shoot, plot_multiple_shoot_multi
export save_node, load_node
export node_loss, eval_node_loss, eval_node_loss_multi
export get_latent_trajectories
export extrapolate_node, load_datasets, encode_datasets, make_nodes
export predictions_and_losses, plot_trajectories, plot_losses
export region_spans!

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - OOD Detection Functions
# ═══════════════════════════════════════════════════════════════════════════════
export fit_knn_ood
export KNN_score
# export KNNOOD

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Normalizer Functions
# ═══════════════════════════════════════════════════════════════════════════════
export normalize_batch, denormalize_batch
export load_normalizer
export compute_normalizer

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Simulation Functions
# ═══════════════════════════════════════════════════════════════════════════════
export circle_shedding_biot, run_sim
export sim_time
export impose_biot_bc_on_snapshot
export flood
export sim_step!
export sim_info

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Custom/Physics Functions
# ═══════════════════════════════════════════════════════════════════════════════
export div_field, div_field_vectorized
export curl_field, curl_vectorized
export velocity_gradient_vectorized
export strain_rate_vectorized, rotation_rate_vectorized
export strain_field
export kinetic_energy_dissipation
export scalar_grad, grad_p
export RHS
export remove_ghosts, remove_buff
export preprocess_data!, clip_time_series

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Data Functions
# ═══════════════════════════════════════════════════════════════════════════════
export build_batch
export get_latent_data
export get_NODE_data
export get_idxs

# Acceleration benchmarking
export AccelResults
export force_stats, record_waterlily_step!, record_prediction!, step_reference!
export compute_metrics, print_metrics
export plot_forces_comparison, plot_timing_bars, plot_accel_combined
export rst_plot, plot_rst_comparison, plot_meanflow_comparison
export save_velocity_frame!, create_velocity_gif, save_accel_plots

# Hybrid simulation
export run_warmup!, run_hybrid!, save_results

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Utilities
# ═══════════════════════════════════════════════════════════════════════════════
export to
export @with_plots, load_plots
export is_hpc, get_device
export cpu_device, gpu_device
export set_seed!
export clear_memory!

# ═══════════════════════════════════════════════════════════════════════════════
# Environment & Device Utilities
# ═══════════════════════════════════════════════════════════════════════════════

"""
    is_hpc()

Check if running on HPC cluster (SLURM, PBS, or LSF).
"""
function is_hpc()
    return haskey(ENV, "SLURM_JOB_ID") || 
           haskey(ENV, "PBS_JOBID") || 
           haskey(ENV, "LSB_JOBID") ||
           get(ENV, "THESIS_HPC", "false") == "true"
end

"""
    get_device(; prefer_gpu=true)

Get the appropriate compute device.
Returns `gpu_device()` if CUDA is available and requested, otherwise `cpu_device()`.
"""
function get_device(; prefer_gpu=true)
    if prefer_gpu && USE_CUDA[]
        return gpu_device()
    else
        return cpu_device()
    end
end

"""
    set_seed!(seed::Int)

Set random seed for reproducibility across all RNGs.
"""
function set_seed!(seed::Int)
    Random.seed!(seed)
    if USE_CUDA[]
        Base.invokelatest(CUDA.seed!, seed)
    end
end

"""
    clear_memory!(; verbose=false)

Force a full GC pass and (if CUDA is loaded) release cached GPU blocks back to
the driver. Use between independent runs in a grid/sweep so VRAM and host RAM
don't accumulate across iterations.
"""
function clear_memory!(; verbose::Bool=false)
    GC.gc(true); GC.gc(true)
    if USE_CUDA[]
        Base.invokelatest(CUDA.reclaim)
    end
    if verbose
        host_free_gb = Sys.free_memory() / 2^30
        host_total_gb = Sys.total_memory() / 2^30
        if USE_CUDA[]
            gpu_free, gpu_total = Base.invokelatest(CUDA.memory_info)
            @info @sprintf("memory: host %.1f/%.1f GB free, gpu %.1f/%.1f GB free",
                host_free_gb, host_total_gb,
                gpu_free / 2^30, gpu_total / 2^30)
        else
            @info @sprintf("memory: host %.1f/%.1f GB free", host_free_gb, host_total_gb)
        end
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Re-exports
# ═══════════════════════════════════════════════════════════════════════════════
export DataLoader
export Chain, Dense, Conv, BatchNorm, MaxPool, Upsample
export relu, tanh, sigmoid, tanhshrink
export WrappedFunction, SamePad
export struct2dict

# JLD2 macros
export @save, @load

end # module Thesis
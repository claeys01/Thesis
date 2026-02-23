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

# Simulations
include("simulations/vortex_shedding_biot_savart.jl")
include("simulations/vortex_shedding.jl")

# Core model definitions
include("AE/AE.jl")
include("AE/AE_train.jl")

# NODE components
include("NODE/NODE_core.jl")
include("NODE/NODE_train.jl")
include("NODE/NODE_PostTrain.jl")



# Combined AE+NODE
include("AENODE.jl")

# Reconstruction utilities
include("utils/AE_reconstructer.jl")
include("utils/AE_loss_plot.jl")

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

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - AE Functions
# ═══════════════════════════════════════════════════════════════════════════════
export train_AE
export load_trained_AE
export visualize_reconstructions
export total_loss, recon_loss, div_loss_L2, masked_loss
export load_simdata

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - NODE Functions
# ═══════════════════════════════════════════════════════════════════════════════
export load_node
export predict_array
export predict_n, predict_n!

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Normalizer Functions
# ═══════════════════════════════════════════════════════════════════════════════
export normalize_batch, denormalize_batch
export load_normalizer
export compute_normalizer

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Simulation Functions
# ═══════════════════════════════════════════════════════════════════════════════
export circle_shedding_biot
export sim_time
export impose_biot_bc_on_snapshot
export flood 

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
export preprocess_data!

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Data Functions
# ═══════════════════════════════════════════════════════════════════════════════
export build_batch
export get_latent_data

# ═══════════════════════════════════════════════════════════════════════════════
# Exports - Utilities
# ═══════════════════════════════════════════════════════════════════════════════
export to  # Just export the TimerOutput instance, not @timeit (already re-exported)
export @with_plots, load_plots
export is_hpc, get_device
export cpu_device, gpu_device

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

export set_seed!

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Re-exports
# ═══════════════════════════════════════════════════════════════════════════════
export DataLoader
export Chain, Dense, Conv, BatchNorm, MaxPool, Upsample
export relu, tanh, sigmoid
export WrappedFunction, SamePad
export struct2dict

# JLD2 macros
export @save, @load

end # module Thesis
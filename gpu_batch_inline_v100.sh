#!/bin/sh
#
#SBATCH --job-name="inline_v100"
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-me-mtt

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export THESIS_HPC="true"
export THESIS_USE_CUDA="true"

# The V100 node's system driver (580 / CUDA 13.0) natively supports this GPU.
# Force CUDA.jl to use that system driver instead of the forward-compatible
# libcuda from CUDA_Driver_jll (reported as 13.1), which fails
# cuDevicePrimaryCtxRetain with error 999 on Volta (sm_70). This is the fix.
export JULIA_CUDA_USE_COMPAT=false

echo "Running on host $(hostname)"
echo "Using $JULIA_NUM_THREADS Julia threads"
echo "Starting at $(date)"

echo "Loading modules"
module purge
module load 2025
module load cuda/12.9
module load cudnn
module load julia
module load slurm
echo "Finished loading modules"

echo "================ GPU DIAGNOSTICS ================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "--- nvidia-smi (DRIVER VERSION is the smoking gun; compare to an A100 node) ---"
nvidia-smi || echo "nvidia-smi not found"
echo "--- /dev/nvidia-uvm (missing => 999 at context creation) ---"
ls -l /dev/nvidia-uvm /dev/nvidiactl /dev/nvidia0 2>&1
echo "--- CUDA.jl view: loaded libcuda path + context retain (all merged to stdout) ---"
julia --project=project -e '
    using CUDA, Libdl
    CUDA.versioninfo()
    println("--- loaded libcuda (system path = good; artifact path = forward-compat) ---")
    foreach(l -> occursin("libcuda", l) && println("  ", l), Libdl.dllist())
    println("functional     = ", CUDA.functional())
    try; println("driver_version = ", CUDA.driver_version()); catch e; println("driver_version FAILED: ", e); end
    print("context retain = ")
    try
        CUDA.context(); println("OK  <<< GPU usable on this node")
    catch e
        println("FAILED 999  <<<"); showerror(stdout, e); println()
    end
' 2>&1 || echo "CUDA probe exited nonzero"
echo "================================================"

echo "Running inline_noload.jl"
srun julia --project=project project/scripts/Inline/inline_noload.jl

echo "Finished at $(date)"

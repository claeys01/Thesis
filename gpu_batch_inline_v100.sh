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
#SBATCH --account=education-ME-msc-mt

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
# NOTE: JULIA_CUDA_USE_BINARYBUILDER is a CUDA.jl 3.x flag and is a no-op in 5.x.
# CUDA.jl uses its own toolkit artifact and ignores `module load cuda`, so the
# only thing that matters here is the node's DRIVER version (see nvidia-smi below).
export THESIS_HPC="true"
export THESIS_USE_CUDA="true"

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
echo "--- CUDA.jl view (artifact CUDA version vs driver, functional, context retain) ---"
julia --project=project -e '
    using CUDA
    CUDA.versioninfo()
    @show CUDA.functional()
    try
        @show CUDA.driver_version()
    catch e
        @warn "driver_version() failed" exception=e
    end
    try
        CUDA.context()            # this is exactly where cuDevicePrimaryCtxRetain (999) fires
        println(">>> context retain OK on this node")
    catch e
        @warn ">>> context retain FAILED (this is the 999)" exception=e
    end
' || echo "CUDA probe exited nonzero"
echo "================================================"

echo "Running inline_noload.jl"
srun julia --project=project project/scripts/Inline/inline_noload.jl

echo "Finished at $(date)"

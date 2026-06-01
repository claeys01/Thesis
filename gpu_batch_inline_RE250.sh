#!/bin/sh
#
#SBATCH --job-name="inline_RE250"
#SBATCH --partition=gpu-a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-me-mtt

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_CUDA_USE_BINARYBUILDER=false
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
echo "Running inline_noload_RE250.jl"

srun julia --project=project project/scripts/Inline/inline_noload_RE250.jl

echo "Finished at $(date)"

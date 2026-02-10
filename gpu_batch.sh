#!/bin/sh
#
#SBATCH --job-name="Lux_AE_gpu"
#SBATCH --partition=gpu-a100
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-ME-msc-mt

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_CUDA_USE_BINARYBUILDER=false

echo "Running on host $(hostname)"
echo "Using $JULIA_NUM_THREADS Julia threads"
echo "Starting at $(date)"

echo "Loading Modules"
module purge
module load 2025
module load cuda/12.9
module load cudnn
module load julia
module load slurm 

echo "Finished Loading Modules"

echo "Running Lux_AE_train.jl"

srun julia --project project/AE/Lux_AE_train.jl

echo "Finished at $(date)"

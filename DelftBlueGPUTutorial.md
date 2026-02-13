# Julia on DelftBlue with CUDA and cuDNN

Building on the existing [software recipe](https://doc.dhpc.tudelft.nl/delftblue/howtos/Julia-with-MPI/) for running Julia scripts on GPUs, this document provides more in-depth tips for running Julia with CUDA enabled on the DelftBlue cluster.

> [!NOTE]
> **Disclaimer:** These are the steps I followed to get Julia running with CUDA and cuDNN on a `gpu-a100` partition for deep learning purposes. This might not be the most optimal or generalisable workflow for other projects. If you only need CUDA (without cuDNN), I recommend following the instructions in the [DelftBlue documentation](https://doc.dhpc.tudelft.nl/delftblue/howtos/Julia-with-MPI/).

---

## Installing CUDA

Install Julia's CUDA package on a **visual node**, since visual nodes have access to both a GPU and the internet. When adding CUDA to Julia, we need to ensure that CUDA always uses the locally installed toolkit. This prevents CUDA from attempting to download artifacts from the internet (which fails on GPU nodes).

Submit the following batch script to install CUDA:

```bash
#!/bin/bash
#SBATCH --partition=visual
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=<research/education>-<faculty>-<department>

module load cuda/12.9 julia

# Install CUDA.jl
# CUDA.set_runtime_version!(local_toolkit=true) ensures that Julia 
# will always use the locally installed CUDA version
julia --project=. -e 'using Pkg; Pkg.add(name="CUDA"); using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
```

> [!NOTE]
> Installing CUDA from the software stack is not advised in the DelftBlue documentation, but it must be done this way for cuDNN to work, since cuDNN requires a local version of CUDA to be present on the system.

> [!IMPORTANT]
> When loading the CUDA module, always load version **12.1 or later**. The default CUDA version (11.6) is not compatible with the default Julia version on DelftBlue.

---

## Testing the CUDA Installation

To test the CUDA installation, request an interactive GPU node using the following command:

```bash
srun --mpi=pmix \
     --job-name="int_gpu_job" \
     --partition=gpu-a100-small \
     --time=01:00:00 \
     --ntasks=1 \
     --cpus-per-task=2 \
     --gpus-per-task=1 \
     --mem-per-cpu=8000mb \
     --account=<research/education>-<faculty>-<department> \
     --pty /bin/bash -il
```

After being granted access to the interactive node, load the relevant modules:

```bash
module load cuda/12.9 julia
```

Now test if CUDA is working correctly by running the following Julia script:

```julia
using CUDA

# Check that CUDA is functional
println("CUDA is functional: ", CUDA.functional())
println(CUDA.versioninfo())
println("Using GPU: ", CUDA.name(CUDA.device()))

# Test device memory allocation and computation
d_a = CuArray([1, 2, 3, 4, 5])
println("Original array: ", d_a)

d_b = d_a .^ 2
println("Squared array:  ", d_b)
```

If this runs without errors, CUDA is configured correctly.

---

## Installing cuDNN

The `cudnn` module can only be loaded on GPU nodes, but the cuDNN Julia package must be downloaded on a node with internet access. Therefore, configuring cuDNN requires two steps:

### Step 1: Install the package on a login node

```bash
module load julia
julia --project=. -e 'using Pkg; Pkg.add(name="cuDNN")'
```

### Step 2: Test on a GPU node

Request an interactive GPU node (see command above), then run the following Julia script:

```julia
using CUDA
using cuDNN

println("CUDA functional: ", CUDA.functional())
println("cuDNN version:   ", cuDNN.version())
println("cuDNN handle:    ", cuDNN.handle())
println("✓ cuDNN is working!")
```

If this does not raise any errors, cuDNN has been configured correctly.

---

## Example Batch Script for GPU Jobs

Once CUDA and cuDNN are installed, you can submit GPU jobs using a batch script. Below is an extensive template you can adapt for your own projects:

```bash
#!/bin/bash
#SBATCH --job-name="my_gpu_job"           # Job name (appears in squeue)
#SBATCH --partition=gpu-a100              # GPU partition (see note below)
#SBATCH --output=logs/%x_%j.out           # Standard output log (%x=job name, %j=job ID)
#SBATCH --error=logs/%x_%j.err            # Standard error log
#SBATCH --time=01:00:00                   # Job runtime (HH:MM:SS)
#SBATCH --ntasks=1                        # Number of tasks (usually 1 for single-GPU jobs)
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --gpus-per-task=1                 # GPUs per task
#SBATCH --mem-per-cpu=4G                  # Memory per CPU core
#SBATCH --account=<research/education>-<faculty>-<department>

# Set Julia to use all allocated CPU threads
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Prevent CUDA from trying to download binaries (use local toolkit instead)
export JULIA_CUDA_USE_BINARYBUILDER=false

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $(hostname)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Julia Threads: $JULIA_NUM_THREADS"
echo "Start Time:    $(date)"
echo "============================================"

module purge                # Clear any previously loaded modules
module load 2025            # Load the 2025 software stack
module load cuda/12.9       # Load CUDA (must match installed version in Julia)
module load cudnn           # Load cuDNN
module load julia           # Load Julia
module load slurm           # Load SLURM utilities

echo "Modules loaded successfully"

# Create logs directory if it doesn't exist
mkdir -p logs

# Run your Julia script
# Adjust the --project path and script path to match your setup
srun julia --project=. path/to/your_script.jl

echo "============================================"
echo "End Time: $(date)"
echo "Job finished"
echo "============================================"
```

### Usage

1. Save this script as `gpu_job.sh` (or any name you prefer)
2. Create a `logs` directory: `mkdir -p logs`
3. Replace `<research/education>-<faculty>-<department>` with your account
4. Replace `path/to/your_script.jl` with your actual script path
5. Submit the job: `sbatch gpu_job.sh`

### Available GPU Partitions

| Partition | GPUs | Max Time | Notes |
|-----------|------|----------|-------|
| `gpu-a100-small` | 1-2 A100 | 24 hours | Smaller jobs, short queue  |
| `gpu-a100` | 1-4 A100 | 120 hours | Standard GPU partition |
| `visual` | 1 | 8 hours | Has internet access (for installation) |

> [!TIP]
> For shorter jobs or debugging, use `gpu-a100-small` (has limited amount of memory) for faster queue times. Use `gpu-a100` for longer training runs.
---

## Summary

| Step | Node Type | Command |
|------|-----------|---------|
| Install CUDA | Visual node | `sbatch install_cuda.sh` |
| Install cuDNN | Login node | `julia -e 'Pkg.add("cuDNN")'` |
| Test CUDA/cuDNN | GPU node | `srun ... --pty /bin/bash` |
| Submit GPU job | Login node | `sbatch gpu_job.sh` |

---

## Troubleshooting

- **CUDA not functional:** Ensure you loaded the CUDA module (`module load cuda/12.9`) before starting Julia.
- **cuDNN version mismatch:** Make sure the CUDA module version matches what cuDNN expects. Using `cuda/12.9` is recommended.
- **Internet access errors on GPU nodes:** This is expected. Always install packages on login or visual nodes first.
- **`local_toolkit=true` not set:** If CUDA tries to download artifacts, re-run `CUDA.set_runtime_version!(local_toolkit=true)` on a visual node.
- **Job stuck in queue:** Try using `gpu-a100-small` partition or reducing requested resources.
- **Out of memory errors:** Increase `--mem-per-cpu` or reduce batch size in your script.
- **Logs directory not found:** Ensure you create the `logs` directory before submitting: `mkdir -p logs`.
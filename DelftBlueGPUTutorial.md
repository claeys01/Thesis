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
     --time=03:00:00 \
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

## Summary

| Step | Node Type | Command |
|------|-----------|---------|
| Install CUDA | Visual node | `sbatch install_cuda.sh` |
| Install cuDNN | Login node | `julia -e 'Pkg.add("cuDNN")'` |
| Test CUDA/cuDNN | GPU node | `srun ... --pty /bin/bash` |

---

## Troubleshooting

- **CUDA not functional:** Ensure you loaded the CUDA module (`module load cuda/12.9`) before starting Julia.
- **cuDNN version mismatch:** Make sure the CUDA module version matches what cuDNN expects. Using `cuda/12.9` is recommended.
- **Internet access errors on GPU nodes:** This is expected. Always install packages on login or visual nodes first.
- **`local_toolkit=true` not set:** If CUDA tries to download artifacts, re-run `CUDA.set_runtime_version!(local_toolkit=true)` on a visual node.
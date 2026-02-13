# Julia on DelftBlue with CUDA and cuDNN tips

Building on the already existing [software recipe](https://doc.dhpc.tudelft.nl/delftblue/howtos/Julia-with-MPI/) for running Julia scripts on GPU's, this document aims to give some more in depth tips for running Julia with CUDA enabled on the DelftBlue Cluster. 

**Disclaimer** These are the steps I followed to get Julia Running with CUDA and cuDNN on a gpu-a100 partition for deep learning purposes. This might not be the most optimal or generalisable workflow for other projects. If you just need CUDA (and not cuDNN), I would advise to follow the instructions listed in the DelftBlue documentation.

### Installing CUDA

Install Julia's CUDA package on visual node, since visual nodes have acces to both a GPU and the internet, by running the following batch script:

```bash
#!/bin/bash
#SBATCH --partition=visual
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:10:00
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=<research/education>-<faculty>-<department>

module load cuda/12.9 julia

# Install CUDA.jl -- This will only work correctly on the "visual" partition,
# because login nodes do not have a GPU and GPU nodes do not have an internet connection
# We then set CUDA to only use the locally installed toolkit to ensure Julia does not download it from the internet.

julia --project=. -e 'using Pkg; Pkg.add(name="CUDA"); using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
```
> [!NOTE]
 Installing CUDA from the software stack not advised in the DelftBlue documentation, but it has to be done this way for cuDNN to work, since it requires a local version of CUDA to be present on the system.  


```bash
srun --mpi=pmix --job-name="int_gpu_job" --partition=gpu-a100-small --time=03:00:00 --ntasks=1 --cpus-per-task=2 --gpus-per-task=1 --mem-per-cpu=8000mb --account=education-<faculty>-<department> --pty /bin/bash -il
```

After having been granted acces to the interactive node, install CUDA and cuDNN. **Note:** when loading the CUDA module, always load a version later than 12.1, since the default cuda version (11.6) is not compatible with the default Julia version on DelftBlue. 

```bash
module load 2025 cuda/12.9 cudnn julia
```
After cuda and cudnn have been loaded into the interactive session, we can go ahead and install CUDA in julia, making sure that the julia package points towards the locally installed cuda version. We need to do this because we do not want julia to 

```bash
julia -e 'using Pkg; Pkg.add(name="CUDA"); using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Julia research project accelerating incompressible CFD (vortex shedding around cylinders) using a hybrid ML approach:
- **AE**: Compress 256×256×4 flow fields → 16-dim latent space
- **NODE**: Learn dynamics dz/dt = f_θ(z) in latent space (Tsit5 solver)
- **KNN OOD detection**: Monitor latent drift against training distribution
- **Hybrid runner**: Replace expensive CFD steps with fast AE+NODE predictions

The main Julia module is `Thesis` in `project/src/Thesis.jl`. All source lives under `project/`.

## Commands

```bash
# Instantiate dependencies
cd project && julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Interactive REPL with project loaded
cd project && julia --project=.

# Run a script
cd project && julia --project=. scripts/AE/train_ae.jl

# Enable GPU
THESIS_USE_CUDA=true julia --project=.
```

HPC (DelftBlue SLURM): `sbatch gpu_batch.sh` / `gpu_batch_node.sh` / `gpu_batch_inline.sh`. Logs go to `logs/`.

## Language & Frameworks

- **Julia only.** Never suggest Python equivalents.
- **Lux** for neural networks — never Flux.
- **CUDA** for GPU — never Reactant. GPU toggled via `ENV["THESIS_USE_CUDA"] = "true"` and `use_gpu` on config structs.
- **JLD2** (`@save`/`@load`) for all model checkpoints — never HDF5, BSON, or JSON.
- **OrdinaryDiffEq + DiffEqFlux** for NODEs. Default solver: `Tsit5()`.
- **Zygote** (`AutoZygote()`) as default AD backend.

## Code Style

- 4-space indentation; `snake_case` functions/variables, `PascalCase` types.
- `Float32` for GPU tensors (`1.0f0`, `1e-8f0`); `Float64` only for ODE tolerances and config structs.
- Greek symbols are idiomatic: `η` (learning rate), `λ` (regularization), `μ`/`σ` (statistics), `μ₀` (base flow).
- Config types use `Base.@kwdef mutable struct` with sensible defaults.
- Mode switches use `Symbol` (`:L1`, `:L2`, `:zscore`, `:charb`).
- No decorative comment blocks, banner headers, or line separators (`# ═══`, `# ---`). Inline comments only where intent isn't obvious. Don't add docstrings or type annotations to code you didn't write.

## Architecture Patterns

**Data flow**: `SimData` / `EpochData` / `LatentData` (defined in `utils/SimDataTypes.jl`) carry raw and processed simulation data through the pipeline.

**Config-driven training**: `LuxArgs` (AE) and `NodeArgs` (NODE) hold all hyperparameters; pass these structs to training functions — never hardcode hyperparameters.

**Model structs**: `NODE`, `AE`, `AENODE`, `Normalizer` are `mutable struct` holding parameters, state, and solver config. Constructors double as factories (e.g., `NODE(latent_dim, dense_mult; ...)`).

**Device handling**: Always use `get_device()` / `cpu_device()` / `gpu_device()` from Lux — never call CUDA directly in model code. CUDA/cuDNN load conditionally in `Thesis.__init__()`.

**Physics losses** in the AE are weighted and summed with reconstruction loss: divergence (`λdiv`), curl (`λcurl`), strain (`λstrain`). Ghost cells from WaterLily are clipped via `clip_bc` before training. Physics operators live in `utils/custom.jl`.

**OOD detection**: `KNNOOD` fits a KDTree on training latent vectors. `predict_flex` in `AENODE.jl` uses KNN scores to flag when retraining is needed.

## Don't

- Don't create wrapper functions for one-time operations.
- Don't add type annotations or docstrings to code you didn't write.
- Don't duplicate — check `src/utils/` and `src/` before writing new helpers.
- Don't add excessive logging beyond what the codebase already uses.

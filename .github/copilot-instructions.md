# Thesis Project — Copilot Instructions

## Project Overview

This is a Julia scientific computing project for ML-enhanced CFD (vortex shedding around cylinders). It combines autoencoders (AE), Neural ODEs (NODE), and out-of-distribution detection (KNN) to accelerate incompressible flow simulations. The main module is `Thesis` in `project/src/Thesis.jl`.

## Language & Framework Rules

- **Julia only.** Never use or suggest Python equivalents.
- **Lux for neural networks.** Never use Flux. The project uses the Lux ecosystem (`Lux`, `LuxCore`, `NNlib`).
- **Float32 for GPU tensors.** Use `Float32` literals (`1.0f0`, `1e-8f0`) in numerical code. Use `Float64` only for ODE tolerances and configuration structs.
- **JLD2 for serialization.** Use `@save`/`@load` macros. Never suggest HDF5, BSON, or JSON for model checkpoints.
- **OrdinaryDiffEq + DiffEqFlux** for Neural ODEs. Default solver is `Tsit5()`.
- **Zygote** is the default AD backend (`AutoZygote()`). Enzyme is available as an alternative.

## Coding Style

- 4-space indentation.
- `snake_case` for functions and variables; `PascalCase` for types/structs.
- Greek symbols are used and expected: `η` (learning rate), `λ` (regularization), `μ`/`σ` (statistics), `μ₀` (base flow).
- Use `Base.@kwdef mutable struct` for configuration types with sensible defaults.
- Use `Symbol` for mode switches (`:L1`, `:L2`, `:zscore`, `:charb`).
- Keep comments minimal — no decorative comment blocks, line separators, or banner-style headers. Inline comments only where intent isn't obvious. Don't add docstrings or type annotations to existing code.

## Architecture Patterns

- Config structs (`LuxArgs`, `NodeArgs`) hold all hyperparameters; pass them to training functions.
- Model structs (`NODE`, `AENODE`, `Normalizer`) are `mutable struct` holding parameters, state, and solver config.
- Constructors double as factory functions (e.g., `NODE(latent_dim, dense_mult; ...)`).
- Data flows through `SimData` / `EpochData` / `LatentData` types defined in `utils/SimDataTypes.jl`.
- Functions return both results and metadata (e.g., `normalize_batch` returns `(normalized_data, normalizer)`).

## GPU / HPC Conventions

- **Always use CUDA for GPU acceleration. Never use or suggest Reactant.**
- GPU use is toggled via `ENV["THESIS_USE_CUDA"] = "true"` and the `use_gpu` field on config structs.
- Always use `get_device()` / `cpu_device()` from Lux for device placement — never call CUDA directly in model code.
- CUDA/cuDNN are loaded conditionally in `Thesis.__init__()`. Assume graceful CPU fallback.
- SLURM batch scripts live at the repo root (`gpu_batch*.sh`). Job logs go to `logs/`.
- Scripts belong in `project/scripts/` (organized by component: `AE/`, `NODE/`, `AE+NODE/`, etc.).
- HPC job output/error logs go in `logs/`.

## Physics-Informed Patterns

- The AE encodes 256×256×4 fields (2 velocity + 2 parameter channels) → 16-dim latent space.
- Physics losses: divergence (`λdiv`), curl (`λcurl`), strain (`λstrain`) are weighted and summed with reconstruction loss.
- Ghost cells from WaterLily are clipped via `clip_bc` before training.
- Normalization is per-channel z-score by default (`:zscore`).

## Don't

- Don't create wrapper functions for one-time operations.
- Don't add type annotations or docstrings to code you didn't write.
- Don't duplicate functionality — check existing utilities in `src/utils/` and `src/` before writing new helpers.
- Don't add decorative comment blocks, banner headers, or line separators (`# ═══`, `# ---`, `# ***`, etc.).
- Don't add excessive logging beyond what the codebase already uses.

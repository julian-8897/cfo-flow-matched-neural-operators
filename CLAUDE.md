# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Interactive marimo notebook demonstrating **CFO** (Continuous-Time PDE Dynamics via Flow Matching, arXiv:2512.05297) for the MoLab competition. The notebook reproduces the Lorenz benchmark from the paper, comparing CFO against autoregressive baselines.

## Commands

```bash
# Run notebook (read-only, browser opens automatically)
uv run marimo run notebook.py

# Edit notebook interactively
uv run marimo edit notebook.py

# Install / sync dependencies
uv sync
```

## Architecture

All logic lives in `notebook.py` — a single marimo app composed of reactive cells. Marimo executes cells as a DAG: a cell's return values are injected as arguments into downstream cells that name them.

**Key cell conventions:**
- Variables prefixed with `_` are cell-local and do not propagate to dependents.
- `hide_code=True` marks presentation-only cells (markdown explainers, static plots).
- `mo.stop(condition, output)` short-circuits a cell and its dependents — used to gate training on button presses.

**Cell dependency graph (high-level):**

```
imports  →  model_definition  →  training_controls / run_training
                ↓                        ↓
         data_controls           error_over_time
                ↓                        ↓
     data_spline_viz          keep_rate_sweep
     data_efficiency_viz      phase_portrait_viz
                              lyapunov_viz
                              physics_interp_viz
```

**`model_definition` cell** — defines and exports all reusable Python objects:
- `TinyODENet`: CFO backbone, learns `du/dt = N_θ(t, u)` with sinusoidal time encoding.
- `ARNet`: autoregressive baseline, predicts `Δu` for one step.
- `generate_lorenz`, `rk4_np`: ground-truth Lorenz trajectory generation.
- `rk4_ode`: integrates `du/dt = model_fn(t, u)` with RK4 at inference.
- `make_cfo_fn`: wraps a trained `TinyODENet` as a `numpy`-compatible `fn(t, u)`.
- `ar_rollout`: autoregressive rollout in normalised space.
- `compute_normalization`: computes per-state-dim mean/std from training trajectories.

**`run_training` cell** — trains four models on button press:
1. **CFO** — cubic spline derivative targets, sparse data (flow-matching loss only).
2. **CFO-PI** — warm-started from CFO, adds physics regularisation: `L = L_flow + λ * L_physics` where `L_physics` penalises deviation of `N_θ(t, u)` from the known Lorenz equations at random attractor points.
3. **AR-full** — 100 % uniform data (paper's hard baseline).
4. **AR-equal** — same sparse kept pairs as CFO.

Returns `cfo_model`, `cfo_pi_model`, `ar_model`, `ar_eq_model`, `norm_stats` for downstream cells.

**Coordinate-space accounting** (`make_cfo_fn` output):
- Input: `u_n` (normalised state), `t_n` (normalised time `t / T_MAX`)
- Output: `d(u_n)/d(t_n)` (normalised velocity)
- To convert to physical velocity: `du_phys/dt_phys = output * state_std / T_MAX`
- Physics loss target: `v_tgt = (v_phys / state_std * T_MAX - du_mean) / du_std`

**Interactive controls** (`data_controls` cell): `keep_rate_slider`, `n_traj_slider`, `horizon_slider` — changing any slider reactively re-runs all dependent visualisation cells without retraining.

## Marimo-specific notes

- Editing notebook cells via the marimo UI writes back to `notebook.py` as plain Python.
- Each `@app.cell` function signature declares its upstream dependencies by name — marimo resolves execution order automatically.
- The notebook targets CPU-only PyTorch (see `pyproject.toml` — pytorch-cpu index).

# CFO: Continuous Vector Fields from Sparse Lorenz Data

[![marimo demo](https://img.shields.io/badge/marimo-demo-blue)](https://molab.marimo.io/notebooks/nb_KjqXxSQBsdQUwggJE6qDh3)

An interactive marimo notebook reproducing the CFO (Continuous Flow Operator) paper on the Lorenz attractor, with a novel parametric extension that learns an entire family of attractors conditioned on the Lorenz parameter.

**Paper**: Hou, Huang & Perdikaris, 2025 — [alphaxiv:2512.05297](https://alphaxiv.org/abs/2512.05297)
**Competition**: alphaXiv · marimo notebook competition

## Live Demo

Run the notebook interactively without installing anything:

**[Open in marimo](https://molab.marimo.io/notebooks/nb_KjqXxSQBsdQUwggJE6qDh3)**

## What This Notebook Shows

### Reproduction
1. **CFO's core advantage**: A continuous-time neural operator trained on sparse, irregular observations outperforms autoregressive baselines that require uniform spacing.
2. **Resolution-agnostic inference**: Because CFO learns a velocity field, it can be queried at any step size without retraining. AR models are locked to their training delta.
3. **Spline-based flow matching**: Quintic Hermite splines provide analytic derivative targets, eliminating the need to differentiate through an ODE solver during training.

### Novel Extension
4. **Parametric CFO**: A single model conditioned on the Lorenz parameter rho learns the entire attractor family at once. At inference, setting rho yields a continuous vector field for that attractor without retraining.
5. **Interactive exploration**: Click anywhere on the parametric vector field heatmap to launch trajectories from arbitrary initial conditions.

## Running Locally

```bash
# Clone the repository
git clone <repo-url>
cd cfo-flow-matched-neural-operators

# Install dependencies (uv recommended)
uv sync

# Run the notebook
marimo run notebook.py

# Or edit in the marimo IDE
marimo edit notebook.py
```

### Dependencies

- Python >= 3.11
- marimo >= 0.10.0
- torch >= 2.2.0 (CPU)
- numpy >= 1.26.0
- matplotlib >= 3.8.0
- scipy >= 1.11.0
- plotly >= 5.20.0

## Notebook Structure

The notebook follows a pedagogical arc:

1. **Theory** — Why autoregressive models fail on sparse/irregular data and how CFO reframes the problem as learning an ODE velocity field.
2. **Mechanism** — Quintic spline interpolation provides flow-matching targets without ODE solver gradients.
3. **Data Pipeline** — Visualise sparse observations, spline reconstruction, and derivative targets at variable keep rates.
4. **Core Experiment** — Train three models simultaneously (CFO, AR-full, AR-equal) and compare RMSE over rollout horizons.
5. **Temporal Generalisation** — Drag a slider to evaluate the same CFO model at arbitrary resolutions.
6. **Parametric Extension** — Train one model on rho in {25, 28, 32, 35, 38}, then generalise across the full interval [15, 50].
7. **Interactive Demo** — Click-to-launch trajectories on the learned parametric vector field.

## Key Takeaways

| # | What this notebook shows | Role |
|---|---|---|
| 1 | CFO reproduces strong performance from sparse, irregular observations against AR baselines | Reproduction |
| 2 | CFO learns a continuous vector field that is resolution-agnostic | Reproduction + demo |
| 3 | Parametric CFO conditions on Lorenz rho, learning the entire family of attractors with one model | Novel |
| 4 | Clickable vector field lets you launch trajectories from any initial condition across the attractor family | Novel demo |

## Limitations

This notebook uses a toy setting (TinyODENet ~5 K params, 3-D Lorenz ODE) to faithfully reproduce the conceptual CFO advantages. The full paper scales to 1D/2D PDE benchmarks with U-Net/FNO operators (200 K–600 K parameters) and achieves up to 87% relative error reduction.

## License

MIT

# Parametric Flow Operators: Learning Families of Chaotic Attractors

[![marimo demo](https://img.shields.io/badge/marimo-Live%20Demo-blue?style=for-the-badge&logo=marimo)](https://molab.marimo.io/notebooks/nb_KjqXxSQBsdQUwggJE6qDh3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> Autoregressive models break when you change the time resolution or when you have sparse observations.
> **Continuous Flow Operators (CFO)** solve this by learning the underlying velocity field of a dynamical system
> instead of a discrete step-map, making them resolution-agnostic and robust to sparse, irregular data.

This project **reproduces the CFO paper** ([Hou, Huang & Perdikaris, 2025](https://alphaxiv.org/abs/2512.05297)) on
the Lorenz attractor, then **extends it** with a novel *Parametric CFO* that learns an entire **family** of attractors
conditioned on system parameters — all from a single model, with no retraining required.

*Built for the [alphaXiv x marimo molab notebook competition](https://marimo.io/pages/events/notebook-competition), 2026.*

---

## Novel Contribution: Parametric CFO

Standard CFO trains one model per fixed system. Here, the network is conditioned on the Lorenz
parameter **ρ**, enabling a *single model* to represent the full family of attractors for **ρ ∈ [15, 50]**.

**Capabilities at inference time:**
- Generalise to **unseen ρ values without retraining**, achieving trajectory accuracy comparable to
  individually-trained fixed-ρ models across the full parameter interval
- Generate a continuous vector field for any specific system configuration on demand
- Explore trajectories interactively — launch from arbitrary initial conditions via a live heatmap

> **[Try it live in the browser →](https://molab.marimo.io/notebooks/nb_KjqXxSQBsdQUwggJE6qDh3)**
>
> *Features: real-time resolution adjustment, trajectory launching, and parametric family exploration.*

---

## Research Reproduction: CFO vs. Autoregressive Models

The core thesis of [Hou, Huang & Perdikaris, 2025](https://alphaxiv.org/abs/2512.05297) is that
learning a velocity field rather than a discrete step-map enables three properties that AR models cannot match:

1. **Sparse & Irregular Data** — By fitting the underlying vector field, CFO remains accurate even with
   aggressive data subsampling where AR models diverge.
2. **Resolution-Agnostic Inference** — The model can be queried at any Δt during inference; standard
   AR models are locked to the training resolution.
3. **Efficient Training** — Quintic Hermite splines provide analytic derivative targets, enabling flow
   matching without differentiating through expensive ODE solvers.

---

## Tech Stack

| Tool | Role |
|---|---|
| [marimo](https://marimo.io/) | Reactive Python notebooks & interactive UI |
| [PyTorch](https://pytorch.org/) | Neural operator training |
| NumPy / SciPy | Spline fitting & numerical integration |
| Plotly / Matplotlib | Visualization |
| [uv](https://github.com/astral-sh/uv) | Dependency management & reproducible environments |

---

## Local Installation

```bash
# Clone the repository
git clone https://github.com/julian-8897/molab-competition.git
cd molab-competition

# Run the interactive notebook
uv run marimo run notebook.py
```

To explore or modify the source in the marimo IDE:

```bash
uv run marimo edit notebook.py
```

---

## License

This project is licensed under the MIT License.

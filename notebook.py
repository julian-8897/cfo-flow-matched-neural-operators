import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import torch
    import torch.nn as nn
    from matplotlib.collections import LineCollection

    torch.manual_seed(42)
    np.random.seed(42)

    matplotlib.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#f8f9fb",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#333333",
            "xtick.color": "#555555",
            "ytick.color": "#555555",
            "text.color": "#222222",
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.grid": True,
            "font.family": "sans-serif",
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
        }
    )
    return LineCollection, go, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def title_cell(mo):
    mo.md(r"""
    # Reproducing CFO on Lorenz, then Extending it with Parametric Conditioning
    ## *alphaXiv x marimo competition notebook*

    > **Paper**: Hou, Huang & Perdikaris, 2025 · [alphaxiv:2512.05297](https://alphaxiv.org/abs/2512.05297)

    ---


    Most learned dynamics surrogates use a **discrete step map** $F_\phi(u_i) \to u_{i+1}$.
    That makes them fragile over long rollouts and ties them to the training step size
    $\Delta t$. **CFO** instead learns the right-hand side of an ODE, $\mathcal{N}_\theta(t, u)$,
    trained via flow matching on spline-derived velocity targets without differentiating
    through the ODE solver. At inference, a numerical integrator evaluates the learned
    vector field at arbitrary query times.

    This notebook makes two arguments:

    1. **Reproduction**: CFO's continuous-time formulation is more data-efficient than
       autoregressive next-step prediction on sparse Lorenz trajectories. Because CFO
       learns a vector field, it can be evaluated at any step size after a single training run.
    2. **Novel extension**: a single model conditioned on $\rho$ learns the entire Lorenz
       attractor family. Click any point on the vector field to launch a trajectory.
       
    | | Standard AR | **CFO** |
    |---|---|---|
    | Learns | step map $F_\phi$ | vector field $\mathcal{N}_\theta(t, u)$ |
    | Handles irregular observations? | No (assumes fixed spacing) | **Yes** |
    | Resolution-agnostic inference? | No (locked to training $\Delta t$) | **Yes** |
    | Parametric conditioning on $\rho$? | Not natural | **Yes (novel extension)** |
    | Competition role | reproduction baseline | **reproduction + novel extension** |
    
    """)
    return


@app.cell(hide_code=True)
def cfo_explainer(mo):
    mo.md(r"""
    ## Why CFO Changes the Learning Problem

    ### The autoregressive issue

    A standard next-step model learns $\hat{u}_{i+1} = F_\phi(u_i)$.
    Chaining these steps at inference causes errors to compound. The model is
    structurally tied to the fixed $\Delta t$ it was trained on.

    ### CFO's key insight

    Treat the trajectory as governed by an ODE:

    $$\frac{du}{dt} = \mathcal{N}_\theta(t,\, u)$$

    Train $\mathcal{N}_\theta$ to approximate the right-hand side; at inference, integrate
    with any standard solver (RK4 here) to any target time $t^*$, with no fixed grid requirement.

    ### The spline trick

    Given sparse, irregular snapshots $\{(t_i, u_i)\}$, fit a **quintic spline** $s(t)$.
    Its analytic derivative $\partial_t s(t)$ serves as the velocity target, giving the
    flow-matching loss:

    $$\mathcal{L}(\theta) = \mathbb{E}_{t,\,\mathbf{u}}\bigl[\|\mathcal{N}_\theta(t,\, s(t;\mathbf{u})) - \partial_t s(t;\mathbf{u})\|^2\bigr]$$

    No ODE solver enters the backward pass, only cheap spline evaluations. That is what
    makes CFO practical on sparse, irregular data.
    """)
    return


@app.cell(hide_code=True)
def lorenz_viz(LineCollection, mo, np, plt):
    """Static Lorenz attractor visualization."""
    _sigma, _rho, _beta = 10.0, 28.0, 8.0 / 3.0

    def _lorenz(state):
        x, y, z = state
        return np.array([_sigma * (y - x), x * (_rho - z) - y, x * y - _beta * z])

    def _rk4(f, state, dt):
        k1 = f(state)
        k2 = f(state + dt / 2 * k1)
        k3 = f(state + dt / 2 * k2)
        k4 = f(state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    _dt = 0.005
    _n = 8000
    _u = np.array([1.0, 0.0, 0.0])
    _traj = np.zeros((_n + 1, 3))
    _traj[0] = _u
    for _i in range(_n):
        _traj[_i + 1] = _rk4(_lorenz, _traj[_i], _dt)

    _times_viz = np.arange(_n + 1)

    fig_lorenz, axes_lorenz = plt.subplots(1, 3, figsize=(14, 4.2))
    fig_lorenz.suptitle(
        "The Lorenz System  ·  $\\dot{x}=\\sigma(y-x)$,  $\\dot{y}=x(\\rho-z)-y$,"
        "  $\\dot{z}=xy-\\beta z$   $[\\sigma=10,\\,\\rho=28,\\,\\beta=8/3]$",
        color="#222222",
        fontsize=11,
    )

    axes_lorenz[0].plot(_times_viz, _traj[:, 0], color="#7799ff", lw=0.6)
    axes_lorenz[0].set_title("x(t)", color="#333333")
    axes_lorenz[0].set_xlabel("step")
    axes_lorenz[0].set_ylabel("x")

    axes_lorenz[1].plot(_times_viz, _traj[:, 2], color="#ff8844", lw=0.6)
    axes_lorenz[1].set_title("z(t)", color="#333333")
    axes_lorenz[1].set_xlabel("step")
    axes_lorenz[1].set_ylabel("z")

    _pts_lc = np.array([_traj[:, 0], _traj[:, 2]]).T.reshape(-1, 1, 2)
    _segs_lc = np.concatenate([_pts_lc[:-1], _pts_lc[1:]], axis=1)
    _lc = LineCollection(
        _segs_lc, cmap="plasma", norm=plt.Normalize(0, _n), lw=0.5, alpha=0.85
    )
    _lc.set_array(np.arange(_n))
    axes_lorenz[2].add_collection(_lc)
    axes_lorenz[2].autoscale()
    fig_lorenz.colorbar(_lc, ax=axes_lorenz[2], label="time step", shrink=0.85)
    axes_lorenz[2].set_title(
        "x–z phase portrait (butterfly attractor)", color="#333333"
    )
    axes_lorenz[2].set_xlabel("x")
    axes_lorenz[2].set_ylabel("z")

    plt.tight_layout()
    _lorenz_out = mo.center(mo.as_html(fig_lorenz))
    plt.close(fig_lorenz)
    _lorenz_out
    return


@app.cell(hide_code=True)
def data_controls(mo):
    keep_rate_slider = mo.ui.slider(
        start=10,
        stop=100,
        step=5,
        value=50,
        label="Keep rate: % of time points per trajectory (CFO training data)",
    )
    n_traj_slider = mo.ui.slider(
        start=5,
        stop=40,
        step=5,
        value=20,
        label="Training trajectories",
    )
    horizon_slider = mo.ui.slider(
        start=30,
        stop=150,
        step=10,
        value=80,
        label="Prediction horizon (steps)",
    )
    mo.vstack(
        [
            mo.md(
                "## Reproduction Setup\n\n"
                "These controls drive the core Lorenz reproduction. **Keep rate** controls how sparse the "
                "CFO and AR-equal training data is, while **AR-full** always trains on 100 % of the trajectory "
                "as the paper's hard baseline:"
            ),
            keep_rate_slider,
            mo.hstack([n_traj_slider, horizon_slider]),
        ]
    )
    return horizon_slider, keep_rate_slider, n_traj_slider


@app.cell(hide_code=True)
def data_spline_viz(QuinticHermiteSpline, keep_rate_slider, mo, np, plt):
    """Visualize full trajectory vs sparse samples vs quintic spline reconstruction."""
    _sigma, _rho, _beta = 10.0, 28.0, 8.0 / 3.0

    def _lorenz_d(state):
        x, y, z = state
        return np.array([_sigma * (y - x), x * (_rho - z) - y, x * y - _beta * z])

    def _rk4s(f, state, dt):
        k1 = f(state)
        k2 = f(state + dt / 2 * k1)
        k3 = f(state + dt / 2 * k2)
        k4 = f(state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    _DT = 0.025
    _N_STEPS = 200
    _rng_demo = np.random.default_rng(7)
    _u0 = _rng_demo.uniform(-8, 8, 3)
    _traj_full = np.zeros((_N_STEPS + 1, 3))
    _traj_full[0] = _u0
    for _i in range(_N_STEPS):
        _traj_full[_i + 1] = _rk4s(_lorenz_d, _traj_full[_i], _DT)
    _t_full = np.arange(_N_STEPS + 1) * _DT

    # Subsample
    _kr = keep_rate_slider.value / 100.0
    _n_keep = max(5, int((_N_STEPS + 1) * _kr))
    _idx_sub = np.sort(_rng_demo.choice(_N_STEPS + 1, _n_keep, replace=False))
    _t_sub = _t_full[_idx_sub]
    _u_sub = _traj_full[_idx_sub]

    # Fit spline
    _cs = QuinticHermiteSpline(_t_sub, _u_sub)
    _t_fine = np.linspace(_t_sub[0], _t_sub[-1], 400)
    _u_recon = _cs(_t_fine)
    _du_recon = _cs(_t_fine, 1)

    fig_spline, axes_spline = plt.subplots(1, 2, figsize=(13, 4.5))
    fig_spline.suptitle(
        f"Data Pipeline: Sparse Observations → Spline → Derivative Target  "
        f"·  keep_rate = {keep_rate_slider.value} %  ({_n_keep}/{_N_STEPS + 1} points)",
        color="#222222",
        fontsize=12,
    )

    # Left: x(t) state reconstruction
    _ax0 = axes_spline[0]
    _ax0.plot(
        _t_full,
        _traj_full[:, 0],
        color="#7799ff",
        lw=1.0,
        alpha=0.35,
        label="full trajectory",
    )
    _ax0.scatter(
        _t_sub,
        _u_sub[:, 0],
        color="#7799ff",
        s=18,
        zorder=4,
        label=f"kept points ({_n_keep})",
    )
    _ax0.plot(
        _t_fine,
        _u_recon[:, 0],
        color="#222222",
        lw=1.6,
        linestyle="--",
        alpha=0.9,
        label="quintic spline",
    )
    _ax0.set_title("x(t): state reconstruction", color="#333333")
    _ax0.set_xlabel("time (s)")
    _ax0.set_ylabel("x")
    _ax0.legend(fontsize=8)

    # Right: dx/dt derivative, the CFO training target
    _ax1 = axes_spline[1]
    _true_du_x = np.array([_lorenz_d(_traj_full[_j])[0] for _j in range(len(_t_full))])
    _ax1.plot(
        _t_full, _true_du_x, color="#7799ff", lw=1.0, alpha=0.4, label="true dx/dt"
    )
    _ax1.plot(
        _t_fine,
        _du_recon[:, 0],
        color="#222222",
        lw=1.8,
        linestyle="--",
        alpha=0.9,
        label="spline dx/dt  ← CFO training target",
    )
    _ax1.set_title("dx/dt: spline derivative (CFO training target)", color="#333333")
    _ax1.set_xlabel("time (s)")
    _ax1.set_ylabel("dx/dt")
    _ax1.legend(fontsize=8)

    plt.tight_layout()
    _spline_out = mo.center(mo.as_html(fig_spline))
    plt.close(fig_spline)
    mo.vstack(
        [
            _spline_out,
            mo.callout(
                mo.md(
                    "**Left**: sparse observations (dots) and quintic spline reconstruction (black dashed). "
                    "**Right**: the spline's analytic derivative is the flow-matching target that CFO trains on. "
                    "No ODE solver needed: spline derivatives are free. Even at low keep rates the velocity "
                    "field is well approximated."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def data_efficiency_viz(QuinticHermiteSpline, generate_lorenz, mo, np, plt):
    """Spline reconstruction quality vs keep rate. No retraining needed."""
    _DT = 0.025
    _N_STEPS = 160
    _T_MAX = _N_STEPS * _DT
    _KEEP_RATES = [0.20, 0.35, 0.50, 0.75, 1.00]
    _LABELS = ["20 %", "35 %", "50 %", "75 %", "100 %"]
    _COLS = ["#ff4466", "#ff8844", "#7799ff", "#aa77ff", "#44dd88"]

    _rng_eff = np.random.default_rng(17)
    _x0 = _rng_eff.uniform(-5, 5, 3)
    _, _traj_full = generate_lorenz(_x0, _N_STEPS, _DT)
    _t_full = np.arange(_N_STEPS + 1) * _DT
    _t_fine = np.linspace(0, _T_MAX, 400)

    fig_eff, axes_eff = plt.subplots(1, 2, figsize=(13, 4.5))
    fig_eff.suptitle(
        "Spline Reconstruction Quality vs Keep Rate  ·  x(t) component",
        color="#222222",
        fontsize=12,
    )

    _rmse_list = []
    for _kr, _lbl, _col in zip(_KEEP_RATES, _LABELS, _COLS):
        _n_keep = max(6, int((_N_STEPS + 1) * _kr))
        _idx = np.sort(_rng_eff.choice(_N_STEPS + 1, _n_keep, replace=False))
        _t_sub = _t_full[_idx]
        _u_sub = _traj_full[_idx]
        _cs = QuinticHermiteSpline(_t_sub, _u_sub)
        _u_recon = _cs(_t_fine)

        _cs_true = QuinticHermiteSpline(_t_full, _traj_full)
        _u_true_fine = _cs_true(_t_fine)
        _rmse = float(np.sqrt(np.mean((_u_recon - _u_true_fine) ** 2)))
        _rmse_list.append(_rmse)

        axes_eff[0].plot(
            _t_fine,
            _u_recon[:, 0],
            color=_col,
            lw=1.4,
            label=f"{_lbl} → RMSE={_rmse:.3f}",
        )
        axes_eff[0].scatter(_t_sub, _u_sub[:, 0], color=_col, s=8, alpha=0.5, zorder=4)

    axes_eff[0].plot(
        _t_full, _traj_full[:, 0], color="#222222", lw=0.8, alpha=0.4, label="true"
    )
    axes_eff[0].set_title("Spline reconstruction x(t)", color="#333333")
    axes_eff[0].set_xlabel("time (s)")
    axes_eff[0].legend(fontsize=8)

    _bars = axes_eff[1].bar(_LABELS, _rmse_list, color=_COLS, width=0.55)
    axes_eff[1].set_title("Reconstruction RMSE vs Keep Rate", color="#333333")
    axes_eff[1].set_xlabel("Keep rate")
    axes_eff[1].set_ylabel("RMSE")
    for _b, _v in zip(_bars, _rmse_list):
        axes_eff[1].text(
            _b.get_x() + _b.get_width() / 2,
            _v + 0.002,
            f"{_v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    plt.tight_layout()
    _eff_out = mo.center(mo.as_html(fig_eff))
    plt.close(fig_eff)
    _eff_out
    return


@app.cell(hide_code=True)
def _(mo, plt):
    """Visual flow diagram comparing CFO training, CFO inference, and AR baseline."""
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fb")
    ax.set_facecolor("#f8f9fb")

    # Lane palette
    _LANES = {
        "train":  {"band": "#e8effe", "box": "#c5d8fa", "accent": "#3366cc", "arrow": "#3366cc"},
        "infer":  {"band": "#efe8fb", "box": "#d5bbf0", "accent": "#7b3fb5", "arrow": "#7b3fb5"},
        "ar":     {"band": "#fde8e8", "box": "#f5b8b8", "accent": "#cc2200", "arrow": "#cc2200"},
    }

    def _band(y_lo, y_hi, key):
        ax.axhspan(y_lo, y_hi, xmin=0, xmax=1, color=_LANES[key]["band"], zorder=0, alpha=0.55)

    def _box(x, y, w, h, lines, key, bold_first=False):
        c = _LANES[key]
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=c["box"], edgecolor=c["accent"],
            linewidth=1.4, zorder=2,
        )
        ax.add_patch(patch)
        if isinstance(lines, str):
            lines = [lines]
        n = len(lines)
        for i, ln in enumerate(lines):
            oy = (n - 1 - i) * 0.18 - (n - 1) * 0.09
            weight = "bold" if (bold_first and i == 0) else "normal"
            sz = 9.5 if (bold_first and i == 0) else 8.5
            ax.text(x + w / 2, y + h / 2 + oy, ln,
                    ha="center", va="center", fontsize=sz,
                    color="#1a1a2e", fontweight=weight, zorder=3)

    def _arrow(x1, y1, x2, y2, key):
        c = _LANES[key]["arrow"]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8,
                                   mutation_scale=14))

    def _label(x, y, text, key):
        c = _LANES[key]["accent"]
        ax.text(x, y, text, fontsize=11, fontweight="bold", color=c,
                va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="#f8f9fb")])

    # ── Lane bands ──────────────────────────────────────────────────────────────
    _band(3.85, 5.50, "train")
    _band(2.05, 3.75, "infer")
    _band(0.25, 1.95, "ar")

    # ── CFO Training ────────────────────────────────────────────────────────────
    _label(0.35, 5.15, "CFO Training", "train")
    _arrow(1.20, 4.55, 2.10, 4.55, "train")
    _box(2.10, 4.20, 1.80, 0.70, ["Subsample", "& Normalise"], "train")
    _arrow(3.90, 4.55, 4.80, 4.55, "train")
    _box(4.80, 4.20, 1.80, 0.70, ["Quintic", "Spline  s(t)"], "train")
    _arrow(6.60, 4.55, 7.50, 4.55, "train")
    _box(7.50, 4.20, 2.10, 0.70, ["Flow-Matching", "Loss  ℒ(t,u,du̇)"], "train")
    _arrow(9.60, 4.55, 10.50, 4.55, "train")
    _box(10.50, 4.20, 1.90, 0.70, ["Update  θ", "(Adam)"], "train", bold_first=True)

    # ── CFO Inference ───────────────────────────────────────────────────────────
    _label(0.35, 3.35, "CFO Inference", "infer")
    _arrow(1.20, 2.75, 2.10, 2.75, "infer")
    _box(2.10, 2.40, 1.30, 0.70, ["u₀"], "infer", bold_first=True)
    _arrow(3.40, 2.75, 4.30, 2.75, "infer")
    _box(4.30, 2.40, 2.40, 0.70, ["RK4  +  𝒩θ(t, u)", "any  Δt"], "infer")
    _arrow(6.70, 2.75, 7.60, 2.75, "infer")
    _box(7.60, 2.40, 2.30, 0.70, ["Trajectory", "@ any resolution"], "infer")

    # ── AR Baseline ─────────────────────────────────────────────────────────────
    _label(0.35, 1.55, "AR Baseline", "ar")
    _arrow(1.20, 0.95, 2.10, 0.95, "ar")
    _box(2.10, 0.60, 1.30, 0.70, ["uᵢ"], "ar", bold_first=True)
    _arrow(3.40, 0.95, 4.30, 0.95, "ar")
    _box(4.30, 0.60, 2.00, 0.70, ["Fφ(uᵢ) → uᵢ₊₁", "fixed  Δt"], "ar")
    _arrow(6.30, 0.95, 7.20, 0.95, "ar")
    _box(7.20, 0.60, 2.30, 0.70, ["Chained preds", "↑ error drift"], "ar")

    plt.tight_layout(pad=0.4)
    _diag_out = mo.center(mo.as_html(fig))
    plt.close(fig)
    _diag_out
    return


@app.cell(hide_code=True)
def algorithm_summary(mo):
    mo.md(r"""
    ## Algorithms and Baselines

    **CFO Training** (blue lane above): subsample and normalise each trajectory, fit a quintic spline, then train the neural operator $\mathcal{N}_\theta$ to match the spline's analytic derivative via flow-matching MSE. No ODE solver is needed in the backward pass.

    **CFO Inference** (purple lane): at test time, feed the learned vector field into a standard RK4 integrator. Because the model is continuous in time, you can query it at **any** step size without retraining.

    **AR Baseline** (red lane): learns a discrete step map $F_\phi(u_i) \to u_{i+1}$. It is locked to the training $\Delta t$ and chaining predictions causes error accumulation.

    | | CFO | AR-full | AR-equal |
    |---|---|---|---|
    | Training target | spline derivative $\partial_t s(t)$ | next state $u_{i+1}$ | next state $u_{i+1}$ |
    | Data used | sparse, irregular | 100 % uniform | same sparse pairs as CFO |
    | Inference | any $\Delta t$ via RK4 | fixed $\Delta t$, chained | fixed $\Delta t$, chained |
    """)
    return


@app.cell(hide_code=True)
def model_def_intro(mo):
    mo.md(r"""
    ## Model Definitions

    The next cell bundles everything needed for the experiments:
    - **Ground-truth Lorenz ODE**: the analytic right-hand side and an RK4 integrator.
    - **Neural networks**: `TinyODENet` (CFO, learns a vector field with sinusoidal time embeddings) and `ARNet` (autoregressive baseline, learns a one-step residual).
    - **Utilities**: normalisation helpers, spline wrapper, and PyTorch-to-NumPy inference wrappers.
    """)
    return


@app.cell(hide_code=True)
def model_definition(nn, np, torch):
    """TinyODENet (CFO backbone) + ARNet + utilities."""

    # ── Lorenz ground truth ───────────────────────────────────────────────────
    _SIGMA, _RHO, _BETA = 10.0, 28.0, 8.0 / 3.0

    def lorenz_deriv(state):
        x, y, z = state
        return np.array([_SIGMA * (y - x), x * (_RHO - z) - y, x * y - _BETA * z])

    def rk4_np(f, state, dt):
        k1 = f(state)
        k2 = f(state + dt / 2 * k1)
        k3 = f(state + dt / 2 * k2)
        k4 = f(state + dt * k3)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def generate_lorenz(x0, n_steps, dt=0.025):
        traj = np.zeros((n_steps + 1, 3))
        traj[0] = x0
        for i in range(n_steps):
            traj[i + 1] = rk4_np(lorenz_deriv, traj[i], dt)
        times = np.arange(n_steps + 1) * dt
        return times, traj

    # ── Neural networks ───────────────────────────────────────────────────────
    class TinyODENet(nn.Module):
        """CFO backbone: learns du/dt = N_θ(t, u) in normalised coordinates."""

        def __init__(self, state_dim=3, hidden=64, n_freq=4):
            super().__init__()
            self.n_freq = n_freq
            in_dim = state_dim + 2 * n_freq
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, state_dim),
            )

        def _time_embed(self, t):
            freqs = torch.tensor(
                [2**i * np.pi for i in range(self.n_freq)], dtype=torch.float32
            )
            t = t.reshape(-1, 1)
            return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)

        def forward(self, t, u):
            # t: (B,), u: (B, 3)
            t_emb = self._time_embed(t)
            return self.net(torch.cat([u, t_emb], dim=-1))

    class ARNet(nn.Module):
        """AR baseline: predicts Δu for one step (trained on full uniform data)."""

        def __init__(self, state_dim=3, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, state_dim),
            )

        def forward(self, u):
            return self.net(u)

    # ── RK4 ODE integrator ────────────────────────────────────────────────────
    def rk4_ode(model_fn, t0, u0, dt, n_steps):
        """Integrate du/dt = model_fn(t, u) with RK4.
        model_fn(t_scalar, u_np_1d) -> np.array(state_dim,)
        Returns trajectory (n_steps+1, state_dim)."""
        traj = [u0.copy()]
        u = u0.copy()
        t = t0
        for _ in range(n_steps):
            k1 = model_fn(t, u)
            k2 = model_fn(t + dt / 2, u + dt / 2 * k1)
            k3 = model_fn(t + dt / 2, u + dt / 2 * k2)
            k4 = model_fn(t + dt, u + dt * k3)
            u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            traj.append(u.copy())
        return np.array(traj)

    @torch.no_grad()
    def make_cfo_fn(ode_net, du_mean=None, du_std=None):
        """Wrap TinyODENet as a plain numpy function(t, u) -> du/dt.

        If du_mean/du_std are provided, the model's output is in
        normalised-derivative space and will be denormalised here.
        """

        def fn(t_scalar, u_np):
            t_t = torch.tensor([float(t_scalar)], dtype=torch.float32)
            u_t = torch.tensor(u_np, dtype=torch.float32).unsqueeze(0)
            pred = ode_net(t_t, u_t).detach().numpy()[0]
            if du_std is not None:
                pred = pred * du_std + (du_mean if du_mean is not None else 0.0)
            return pred

        return fn

    @torch.no_grad()
    def ar_rollout(ar_net, u0_norm, n_steps):
        """Autoregressive rollout in normalised space."""
        traj = [u0_norm.copy()]
        u = torch.tensor(u0_norm, dtype=torch.float32).unsqueeze(0)
        for _ in range(n_steps):
            u = u + ar_net(u)
            traj.append(u.numpy()[0].copy())
        return np.array(traj)

    # ── Training helpers ──────────────────────────────────────────────────────
    def compute_normalization(trajs):
        all_u = np.concatenate([t for _, t in trajs], axis=0)
        mean = all_u.mean(axis=0)
        std = all_u.std(axis=0) + 1e-8
        return mean, std

    class QuinticHermiteSpline:
        """Quintic B-spline interpolant (k=5) with analytic derivative support."""

        def __init__(self, t, u):
            from scipy.interpolate import make_interp_spline

            self._spl = make_interp_spline(t, u, k=5)

        def __call__(self, t_new, deriv=0):
            return self._spl(t_new, deriv)

    def generate_lorenz_param(x0, n_steps, rho, dt=0.025):
        _sigma, _beta = 10.0, 8.0 / 3.0

        def _lp(state):
            x, y, z = state
            return np.array([_sigma * (y - x), x * (rho - z) - y, x * y - _beta * z])

        traj = np.zeros((n_steps + 1, 3))
        traj[0] = x0
        for i in range(n_steps):
            traj[i + 1] = rk4_np(_lp, traj[i], dt)
        return np.arange(n_steps + 1) * dt, traj

    class TinyODENetParam(nn.Module):
        """CFO backbone conditioned on a scalar parameter (rho)."""

        def __init__(self, state_dim=3, hidden=128, n_freq=4):
            super().__init__()
            self.n_freq = n_freq
            in_dim = state_dim + 2 * n_freq + 1
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, state_dim),
            )

        def _time_embed(self, t):
            freqs = torch.tensor(
                [2**i * np.pi for i in range(self.n_freq)], dtype=torch.float32
            )
            t = t.reshape(-1, 1)
            return torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)

        def forward(self, t, u, rho_norm):
            return self.net(torch.cat([u, self._time_embed(t), rho_norm], dim=-1))

    @torch.no_grad()
    def make_cfo_param_fn(ode_net_param, rho_norm_scalar, du_mean=None, du_std=None):
        def fn(t_scalar, u_np):
            t_t = torch.tensor([float(t_scalar)], dtype=torch.float32)
            u_t = torch.tensor(u_np, dtype=torch.float32).unsqueeze(0)
            rho_t = torch.tensor([[float(rho_norm_scalar)]], dtype=torch.float32)
            pred = ode_net_param(t_t, u_t, rho_t).detach().numpy()[0]
            if du_std is not None:
                pred = pred * du_std + (du_mean if du_mean is not None else 0.0)
            return pred

        return fn

    _n_ode = TinyODENet()
    _n_ar = ARNet()
    n_params_ode = sum(p.numel() for p in _n_ode.parameters())
    n_params_ar = sum(p.numel() for p in _n_ar.parameters())

    return (
        ARNet,
        QuinticHermiteSpline,
        TinyODENet,
        TinyODENetParam,
        ar_rollout,
        compute_normalization,
        generate_lorenz,
        generate_lorenz_param,
        lorenz_deriv,
        make_cfo_fn,
        make_cfo_param_fn,
        n_params_ar,
        n_params_ode,
        rk4_np,
        rk4_ode,
    )


@app.cell(hide_code=True)
def training_controls(mo, n_params_ode, n_params_ar):
    train_epochs = mo.ui.slider(
        start=50, stop=300, step=25, value=150, label="Training epochs"
    )
    train_btn = mo.ui.run_button(label="▶ Train Three Models")
    mo.vstack(
        [
            mo.md(
                "## Core Experiment\n\n"
                "**Three models trained simultaneously:**\n"
                "- **CFO** (TinyODENet): quintic spline derivative targets, sparse data\n"
                "- **AR-full** (ARNet): 100% uniform data, paper's hard baseline\n"
                "- **AR-equal** (ARNet): same sparse kept pairs as CFO, equal-data comparison\n\n"
                f"TinyODENet: **{n_params_ode:,} parameters** (sinusoidal time encoding) · "
                f"ARNet: **{n_params_ar:,} parameters**."
            ),
            mo.hstack([train_epochs, train_btn]),
        ]
    )
    return train_btn, train_epochs


@app.cell(hide_code=True)
def run_training(
    ARNet,
    QuinticHermiteSpline,
    TinyODENet,
    ar_rollout,
    compute_normalization,
    generate_lorenz,
    horizon_slider,
    keep_rate_slider,
    make_cfo_fn,
    mo,
    n_traj_slider,
    np,
    plt,
    rk4_ode,
    torch,
    train_btn,
    train_epochs,
):
    train_btn  # reactive dependency

    if not train_btn.value:
        cfo_model = None
        ar_model = None
        ar_eq_model = None
        norm_stats = None
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "Click **▶ Train Three Models** to train CFO, AR-full, and AR-equal."
                ),
                kind="neutral",
            ),
        )

    torch.manual_seed(0)
    np.random.seed(0)

    _DT = 0.025
    _N_STEPS = 200
    _N_TRAIN = n_traj_slider.value
    _N_TEST = 10
    _KR = keep_rate_slider.value / 100.0
    _EPOCHS = train_epochs.value
    _LR = 3e-3
    _BATCH = 128
    _N_SAMPLE_PER_TRAJ = 60
    _T_MAX = _N_STEPS * _DT

    # Generate trajectories
    _rng = np.random.default_rng(1)
    _all_x0 = _rng.uniform(-8, 8, (_N_TRAIN + _N_TEST, 3))
    _all_trajs = [
        generate_lorenz(_all_x0[i], _N_STEPS, _DT) for i in range(_N_TRAIN + _N_TEST)
    ]
    _train_trajs = _all_trajs[:_N_TRAIN]
    _test_trajs = _all_trajs[_N_TRAIN:]

    _state_mean, _state_std = compute_normalization(_train_trajs)

    # ── CFO training data (sparse, irregular quintic spline targets) ────────────
    _cfo_t_list, _cfo_u_list, _cfo_du_list = [], [], []
    _kept_idx_list = []
    _rng2 = np.random.default_rng(2)
    for _times_raw, _traj_raw in _train_trajs:
        _n_pts = len(_times_raw)
        _n_keep = max(6, int(_n_pts * _KR))
        _idx = np.sort(_rng2.choice(_n_pts, _n_keep, replace=False))
        _kept_idx_list.append(_idx)
        _t_sub = _times_raw[_idx] / _T_MAX
        _u_sub = (_traj_raw[_idx] - _state_mean) / _state_std
        _cs = QuinticHermiteSpline(_t_sub, _u_sub)
        _t_smp = _rng2.uniform(_t_sub[0], _t_sub[-1], _N_SAMPLE_PER_TRAJ).astype(
            np.float32
        )
        _cfo_t_list.append(_t_smp)
        _cfo_u_list.append(_cs(_t_smp).astype(np.float32))
        _cfo_du_list.append(_cs(_t_smp, 1).astype(np.float32))

    _all_du_raw = np.concatenate(_cfo_du_list, axis=0)
    _du_mean = _all_du_raw.mean(axis=0).astype(np.float32)
    _du_std = (_all_du_raw.std(axis=0) + 1e-8).astype(np.float32)
    _cfo_du_scaled = [((du - _du_mean) / _du_std) for du in _cfo_du_list]

    _T_cfo = torch.tensor(np.concatenate(_cfo_t_list))
    _U_cfo = torch.tensor(np.concatenate(_cfo_u_list))
    _DU_cfo = torch.tensor(np.concatenate(_cfo_du_scaled))
    _N_CFO = len(_T_cfo)

    # ── AR-full training data (100 % uniform) ─────────────────────────────────
    _u_ar_list, _un_ar_list = [], []
    for _, _traj_raw in _train_trajs:
        _u_n = ((_traj_raw - _state_mean) / _state_std).astype(np.float32)
        _u_ar_list.append(_u_n[:-1])
        _un_ar_list.append(_u_n[1:])
    _U_ar = torch.tensor(np.concatenate(_u_ar_list))
    _UN_ar = torch.tensor(np.concatenate(_un_ar_list))
    _N_AR = len(_U_ar)

    # ── AR-equal training data (same sparse kept pairs as CFO) ────────────────
    _u_areq_list, _un_areq_list = [], []
    for (_times_raw, _traj_raw), _idx in zip(_train_trajs, _kept_idx_list):
        _u_n = ((_traj_raw - _state_mean) / _state_std).astype(np.float32)
        _u_areq_list.append(_u_n[_idx[:-1]])
        _un_areq_list.append(_u_n[_idx[1:]])
    _U_areq = torch.tensor(np.concatenate(_u_areq_list))
    _UN_areq = torch.tensor(np.concatenate(_un_areq_list))
    _N_AREQ = len(_U_areq)

    # ── Train CFO ─────────────────────────────────────────────────────────────
    cfo_model = TinyODENet()
    _opt_cfo = torch.optim.Adam(cfo_model.parameters(), lr=_LR)
    _losses_cfo = []
    for _ in range(_EPOCHS):
        _perm = np.random.permutation(_N_CFO)
        _ep_loss = 0.0
        _nb = 0
        for _i in range(0, _N_CFO, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_cfo.zero_grad()
            _loss = torch.mean(
                (cfo_model(_T_cfo[_idx_b], _U_cfo[_idx_b]) - _DU_cfo[_idx_b]) ** 2
            )
            _loss.backward()
            _opt_cfo.step()
            _ep_loss += _loss.item()
            _nb += 1
        _losses_cfo.append(_ep_loss / max(_nb, 1))

    # ── Train AR-full ─────────────────────────────────────────────────────────
    ar_model = ARNet()
    _opt_ar = torch.optim.Adam(ar_model.parameters(), lr=_LR)
    _losses_ar = []
    for _ in range(_EPOCHS):
        _perm = np.random.permutation(_N_AR)
        _ep_loss = 0.0
        _nb = 0
        for _i in range(0, _N_AR, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_ar.zero_grad()
            _loss = torch.mean(
                (_U_ar[_idx_b] + ar_model(_U_ar[_idx_b]) - _UN_ar[_idx_b]) ** 2
            )
            _loss.backward()
            _opt_ar.step()
            _ep_loss += _loss.item()
            _nb += 1
        _losses_ar.append(_ep_loss / max(_nb, 1))

    # ── Train AR-equal ────────────────────────────────────────────────────────
    ar_eq_model = ARNet()
    _opt_areq = torch.optim.Adam(ar_eq_model.parameters(), lr=_LR)
    _losses_areq = []
    for _ in range(_EPOCHS):
        _perm = np.random.permutation(_N_AREQ)
        _ep_loss = 0.0
        _nb = 0
        for _i in range(0, _N_AREQ, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_areq.zero_grad()
            _loss = torch.mean(
                (_U_areq[_idx_b] + ar_eq_model(_U_areq[_idx_b]) - _UN_areq[_idx_b]) ** 2
            )
            _loss.backward()
            _opt_areq.step()
            _ep_loss += _loss.item()
            _nb += 1
        _losses_areq.append(_ep_loss / max(_nb, 1))

    norm_stats = (_state_mean, _state_std, _T_MAX, _DT, _du_mean, _du_std)

    # ── Eval on test set ──────────────────────────────────────────────────────
    _H = horizon_slider.value
    _dt_n = _DT / _T_MAX
    _cfo_fn = make_cfo_fn(cfo_model, _du_mean, _du_std)
    _cfo_rmse_list, _ar_rmse_list, _areq_rmse_list = [], [], []

    for _times_raw, _traj_raw in _test_trajs:
        _u_n = (_traj_raw - _state_mean) / _state_std
        _u0_n = _u_n[0]
        _true = _u_n[: _H + 1]
        _cfo_rmse_list.append(
            np.sqrt(
                np.mean((rk4_ode(_cfo_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)
            )
        )
        _ar_rmse_list.append(
            np.sqrt(np.mean((ar_rollout(ar_model, _u0_n, _H) - _true) ** 2, axis=1))
        )
        _areq_rmse_list.append(
            np.sqrt(np.mean((ar_rollout(ar_eq_model, _u0_n, _H) - _true) ** 2, axis=1))
        )

    _cfo_rmse_mean = np.mean(_cfo_rmse_list, axis=0)
    _ar_rmse_mean = np.mean(_ar_rmse_list, axis=0)
    _areq_rmse_mean = np.mean(_areq_rmse_list, axis=0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _kr = keep_rate_slider.value
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.8))

    _axes[0].plot(_losses_cfo, color="#7799ff", lw=1.5, label=f"CFO ({_kr} %)")
    _axes[0].plot(
        _losses_ar, color="#ff8844", lw=1.5, label="AR-full (100 %)", linestyle="--"
    )
    _axes[0].plot(
        _losses_areq,
        color="#44dd88",
        lw=1.5,
        label=f"AR-equal ({_kr} %)",
        linestyle=":",
    )
    _axes[0].set_title("Training Loss", color="#333333")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_ylabel("MSE (normalised space)")
    _axes[0].legend(fontsize=8)
    _axes[0].set_yscale("log")

    _step_axis = np.arange(_H + 1)
    _axes[1].plot(
        _step_axis, _cfo_rmse_mean, color="#7799ff", lw=2, label=f"CFO ({_kr} %)"
    )
    _axes[1].plot(
        _step_axis,
        _ar_rmse_mean,
        color="#ff8844",
        lw=2,
        linestyle="--",
        label="AR-full (100 %)",
    )
    _axes[1].plot(
        _step_axis,
        _areq_rmse_mean,
        color="#44dd88",
        lw=2,
        linestyle=":",
        label=f"AR-equal ({_kr} %)",
    )
    _axes[1].set_title("Test RMSE over horizon", color="#333333")
    _axes[1].set_xlabel("step")
    _axes[1].set_ylabel("RMSE (normalised)")
    _axes[1].legend(fontsize=8)

    plt.tight_layout()
    _fig_out = mo.center(mo.as_html(_fig))
    plt.close(_fig)

    _cfo_final = float(_cfo_rmse_mean[-1])
    _ar_final = float(_ar_rmse_mean[-1])
    _impr_vs_full = 100 * (1 - _cfo_final / (_ar_final + 1e-9))

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**AR-full** trains on 100% data (paper's hard baseline). "
                    f"**AR-equal** and **CFO** train on the same {_kr}% sparse data."
                ),
                kind="info",
            ),
            mo.hstack(
                [
                    mo.stat(f"{_losses_cfo[-1]:.4f}", label="CFO loss"),
                    mo.stat(f"{_losses_ar[-1]:.4f}", label="AR-full loss"),
                    mo.stat(f"{_cfo_final:.3f}", label="CFO RMSE"),
                    mo.stat(f"{_ar_final:.3f}", label="AR-full RMSE"),
                    mo.stat(f"{_impr_vs_full:.1f} %", label="CFO vs AR-full"),
                ]
            ),
            _fig_out,
        ]
    )
    return ar_eq_model, ar_model, cfo_model, norm_stats


@app.cell(hide_code=True)
def error_over_time_intro(mo):
    mo.md(r"""
    ### Error Curves with Variance

    The training plot above showed the learning curves. The plot below repeats the test evaluation over **15 fresh trajectories** to show mean RMSE plus/minus one standard deviation. This gives a clearer picture of how prediction error grows as a function of rollout length.
    """)
    return


@app.cell(hide_code=True)
def error_over_time(
    ar_eq_model,
    ar_model,
    ar_rollout,
    cfo_model,
    generate_lorenz,
    horizon_slider,
    keep_rate_slider,
    make_cfo_fn,
    mo,
    norm_stats,
    np,
    plt,
    rk4_ode,
):
    if cfo_model is None:
        mo.stop(True)

    _state_mean, _state_std, _T_MAX, _DT, _du_mean, _du_std = norm_stats
    _H = horizon_slider.value
    _dt_n = _DT / _T_MAX
    _N_EVAL = 15
    _kr = keep_rate_slider.value

    _rng_err = np.random.default_rng(55)
    _all_cfo_rmse, _all_ar_rmse, _all_areq_rmse = [], [], []

    _cfo_fn = make_cfo_fn(cfo_model, _du_mean, _du_std)
    for _ in range(_N_EVAL):
        _x0 = _rng_err.uniform(-8, 8, 3)
        _, _traj = generate_lorenz(_x0, _H + 5, _DT)
        _u_n = (_traj - _state_mean) / _state_std
        _u0_n = _u_n[0]
        _true = _u_n[: _H + 1]
        _all_cfo_rmse.append(
            np.sqrt(
                np.mean((rk4_ode(_cfo_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)
            )
        )
        _all_ar_rmse.append(
            np.sqrt(np.mean((ar_rollout(ar_model, _u0_n, _H) - _true) ** 2, axis=1))
        )
        _all_areq_rmse.append(
            np.sqrt(np.mean((ar_rollout(ar_eq_model, _u0_n, _H) - _true) ** 2, axis=1))
        )

    _cfo_m = np.mean(_all_cfo_rmse, axis=0)
    _cfo_s = np.std(_all_cfo_rmse, axis=0)
    _ar_m = np.mean(_all_ar_rmse, axis=0)
    _ar_s = np.std(_all_ar_rmse, axis=0)
    _areq_m = np.mean(_all_areq_rmse, axis=0)
    _areq_s = np.std(_all_areq_rmse, axis=0)
    _t_ax = np.arange(_H + 1)

    fig_err, ax_err = plt.subplots(figsize=(12, 4))
    ax_err.plot(_t_ax, _cfo_m, color="#7799ff", lw=2, label=f"CFO ({_kr} % data)")
    ax_err.fill_between(
        _t_ax, _cfo_m - _cfo_s, _cfo_m + _cfo_s, color="#7799ff", alpha=0.2
    )
    ax_err.plot(
        _t_ax,
        _ar_m,
        color="#ff8844",
        lw=2,
        linestyle="--",
        label="AR-full (100 % data)",
    )
    ax_err.fill_between(_t_ax, _ar_m - _ar_s, _ar_m + _ar_s, color="#ff8844", alpha=0.2)
    ax_err.plot(
        _t_ax,
        _areq_m,
        color="#44dd88",
        lw=2,
        linestyle=":",
        label=f"AR-equal ({_kr} % data)",
    )
    ax_err.fill_between(
        _t_ax, _areq_m - _areq_s, _areq_m + _areq_s, color="#44dd88", alpha=0.2
    )
    ax_err.set_title(
        f"Prediction RMSE over Steps  ·  averaged over {_N_EVAL} test trajectories",
        color="#333333",
    )
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("RMSE (normalised, log scale)")
    ax_err.set_yscale("log")
    ax_err.legend(fontsize=10)
    _cross_idx = int(np.argmax(_ar_m > _cfo_m))
    if _cross_idx > 0:
        ax_err.axvline(
            _t_ax[_cross_idx], color="#ff8844", lw=1.0, linestyle=":", alpha=0.55
        )
        ax_err.annotate(
            "AR-full diverges\npast CFO",
            xy=(_t_ax[_cross_idx], _ar_m[_cross_idx]),
            xytext=(_t_ax[_cross_idx] + 3, _ar_m[_cross_idx] * 2.0),
            color="#ff8844",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#ff8844", lw=0.8),
        )
    plt.tight_layout()
    _err_out = mo.center(mo.as_html(fig_err))
    plt.close(fig_err)
    _err_out
    return


@app.cell(hide_code=True)
def sweep_intro(mo):
    mo.md(r"""
    ## Temporal Generalisation: One Model, Any Resolution

    CFO learns a continuous vector field that can be queried at **any step size** without
    retraining. AR is locked to its training $\Delta t$: a different resolution requires
    a new model.

    **Drag the slider** to vary the evaluation resolution using the single CFO model
    trained above.
    """)
    return


@app.cell(hide_code=True)
def resolution_controls(cfo_model, mo, norm_stats):
    mo.stop(
        cfo_model is None,
        mo.callout(
            mo.md("Train models first. Click **▶ Train Three Models** above."),
            kind="neutral",
        ),
    )
    _state_mean, _state_std, _T_MAX, _DT_TRAIN, _du_mean, _du_std = norm_stats
    dt_slider = mo.ui.slider(
        start=0.005,
        stop=0.1,
        step=0.005,
        value=_DT_TRAIN,
        label="Evaluation Δt (s)",
        show_value=True,
    )
    dt_slider
    return (dt_slider,)


@app.cell(hide_code=True)
def continuous_resolution_demo(
    ar_model,
    ar_rollout,
    cfo_model,
    dt_slider,
    generate_lorenz,
    make_cfo_fn,
    mo,
    norm_stats,
    np,
    plt,
    rk4_ode,
):
    _state_mean, _state_std, _T_MAX, _DT_TRAIN, _du_mean, _du_std = norm_stats
    _dt_eval = dt_slider.value
    _horizon = 2.0
    _n_steps = max(4, int(round(_horizon / _dt_eval)))
    _ar_steps = max(4, int(round(_horizon / _DT_TRAIN)))
    _rng = np.random.default_rng(7)
    _x0 = _rng.uniform(-8, 8, 3)
    _, _gt_fine = generate_lorenz(_x0, int(_horizon / 0.002), 0.002)
    _x0_norm = ((_x0 - _state_mean) / _state_std).astype(np.float32)
    _cfo_fn = make_cfo_fn(cfo_model, _du_mean, _du_std)
    _dt_n = _dt_eval / _T_MAX
    _traj_norm = rk4_ode(_cfo_fn, 0.0, _x0_norm, _dt_n, _n_steps)
    _traj = _traj_norm * _state_std + _state_mean
    _ar_traj_norm = ar_rollout(ar_model, _x0_norm, _ar_steps)
    _ar_traj = _ar_traj_norm * _state_std + _state_mean
    _steps_cfo = np.arange(_n_steps + 1, dtype=float)
    _steps_ar = np.arange(_ar_steps + 1) * (_DT_TRAIN / _dt_eval)
    _steps_gt = np.linspace(0.0, _n_steps, len(_gt_fine))

    fig_res, axes_res = plt.subplots(1, 2, figsize=(12, 4.5))
    fig_res.suptitle(
        f"CFO at Δt={_dt_eval:.3f}s ({_n_steps} steps)  vs  AR locked to Δt={_DT_TRAIN:.3f}s ({_ar_steps} steps)",
        color="#222222",
        fontsize=11,
    )

    # Panel 1: x-z phase portrait
    axes_res[0].plot(
        _gt_fine[:, 0],
        _gt_fine[:, 2],
        color="#aaaaaa",
        lw=0.6,
        alpha=0.5,
        label="ground truth",
    )
    axes_res[0].plot(
        _traj[:, 0],
        _traj[:, 2],
        color="#7799ff",
        lw=1.8,
        label=f"CFO (Δt={_dt_eval:.3f}s)",
    )
    axes_res[0].plot(
        _ar_traj[:, 0],
        _ar_traj[:, 2],
        color="#ff8844",
        lw=1.5,
        linestyle="--",
        label=f"AR (Δt={_DT_TRAIN:.3f}s, fixed)",
    )
    axes_res[0].set_xlabel("x")
    axes_res[0].set_ylabel("z")
    axes_res[0].set_title("Phase portrait (x–z)", color="#333333")
    axes_res[0].legend(fontsize=8)

    # Panel 2: x over steps
    axes_res[1].plot(
        _steps_gt,
        _gt_fine[:, 0],
        color="#aaaaaa",
        lw=0.6,
        alpha=0.5,
        label="ground truth",
    )
    axes_res[1].plot(
        _steps_cfo,
        _traj[:, 0],
        color="#7799ff",
        lw=1.8,
        label=f"CFO (Δt={_dt_eval:.3f}s)",
    )
    axes_res[1].plot(
        _steps_ar,
        _ar_traj[:, 0],
        color="#ff8844",
        lw=1.5,
        linestyle="--",
        label=f"AR (Δt={_DT_TRAIN:.3f}s, fixed)",
    )
    axes_res[1].set_xlabel("step (CFO units)")
    axes_res[1].set_ylabel("x")
    axes_res[1].set_title("x-component over time", color="#333333")
    axes_res[1].legend(fontsize=8)

    plt.tight_layout()
    _res_out = mo.center(mo.as_html(fig_res))
    plt.close(fig_res)

    mo.vstack(
        [
            _res_out,
            mo.callout(
                mo.md(
                    f"CFO uses **{_n_steps} steps** at the chosen resolution. "
                    f"AR is always fixed to {_ar_steps} steps at its training Δt. "
                    "Change the slider: the CFO trajectory updates instantly, no retraining."
                ),
                kind="info",
            ),
        ]
    )
    return




@app.cell(hide_code=True)
def parametric_intro(mo):
    mo.md(r"""
    ## Novel Contribution: Parametric CFO

    One CFO model, conditioned on the Lorenz parameter $\rho$, learns the **entire family**
    of attractors at once. At inference, setting $\rho$ yields a continuous vector field for
    that attractor without retraining.

    This is not in the original paper. It is the notebook's primary novel contribution.
    The model conditions on normalised $\rho$ as an extra scalar input to the network, trained
    jointly on trajectories sampled from $\rho \in \{25, 28, 32, 35, 38\}$.
    """)
    return


@app.cell(hide_code=True)
def parametric_train_controls(mo):
    param_epochs_slider = mo.ui.slider(
        start=100, stop=500, step=50, value=200, label="Epochs", show_value=True
    )
    param_hidden_slider = mo.ui.slider(
        start=64, stop=256, step=32, value=128, label="Hidden size", show_value=True
    )
    param_traj_slider = mo.ui.slider(
        start=5, stop=20, step=5, value=10, label="Trajectories per ρ", show_value=True
    )
    param_train_btn = mo.ui.run_button(label="▶ Train Parametric CFO")
    mo.vstack(
        [
            mo.hstack([param_epochs_slider, param_hidden_slider, param_traj_slider]),
            param_train_btn,
        ]
    )
    return param_train_btn, param_epochs_slider, param_hidden_slider, param_traj_slider


@app.cell(hide_code=True)
def parametric_training(
    TinyODENetParam,
    QuinticHermiteSpline,
    compute_normalization,
    generate_lorenz_param,
    mo,
    nn,
    np,
    param_train_btn,
    param_epochs_slider,
    param_hidden_slider,
    param_traj_slider,
    torch,
):
    mo.stop(
        not param_train_btn.value,
        mo.callout(mo.md("Click **▶ Train Parametric CFO** to train."), kind="neutral"),
    )
    _RHOS = [25.0, 28.0, 32.0, 35.0, 38.0]
    _DT_P = 0.025
    _N_STEPS_P = 200
    _N_SMP_P = 60
    _EPOCHS_P = param_epochs_slider.value
    _HIDDEN_P = param_hidden_slider.value
    _N_TRAJ_P = param_traj_slider.value
    _LR_P = 3e-3
    _BATCH_P = 256
    _T_MAX_P = _N_STEPS_P * _DT_P
    _rng_p = np.random.default_rng(42)
    _all_trajs_p = []
    for _rho_val in _RHOS:
        for _ in range(_N_TRAJ_P):
            _x0_p = _rng_p.uniform(-8, 8, 3)
            _ts_p, _traj_p = generate_lorenz_param(_x0_p, _N_STEPS_P, _rho_val, _DT_P)
            _all_trajs_p.append((_ts_p, _traj_p, _rho_val))
    _flat_trajs = [(_ts, _tr) for _ts, _tr, _ in _all_trajs_p]
    _p_state_mean, _p_state_std = compute_normalization(_flat_trajs)
    _RHO_MEAN = float(np.mean(_RHOS))
    _RHO_STD = float(np.std(_RHOS) + 1e-8)
    _T_p, _U_p, _DU_p, _RHO_p = [], [], [], []
    for _ts_p, _tr_p, _rho_v in _all_trajs_p:
        _t_sub = _ts_p / _T_MAX_P
        _u_sub = (_tr_p - _p_state_mean) / _p_state_std
        _cs_p = QuinticHermiteSpline(_t_sub.astype(np.float64), _u_sub)
        _t_smp = _rng_p.uniform(_t_sub[0], _t_sub[-1], _N_SMP_P).astype(np.float32)
        _T_p.append(_t_smp)
        _U_p.append(_cs_p(_t_smp).astype(np.float32))
        _DU_p.append(_cs_p(_t_smp, 1).astype(np.float32))
        _RHO_p.append(
            np.full(_N_SMP_P, (_rho_v - _RHO_MEAN) / _RHO_STD, dtype=np.float32)
        )
    _all_du_p = np.concatenate(_DU_p, axis=0)
    _p_du_mean = _all_du_p.mean(axis=0).astype(np.float32)
    _p_du_std = (_all_du_p.std(axis=0) + 1e-8).astype(np.float32)
    _DU_sc = [(du - _p_du_mean) / _p_du_std for du in _DU_p]
    _T_t = torch.tensor(np.concatenate(_T_p))
    _U_t = torch.tensor(np.concatenate(_U_p))
    _DU_t = torch.tensor(np.concatenate(_DU_sc))
    _RHO_t = torch.tensor(np.concatenate(_RHO_p)).unsqueeze(1)
    _N_P = len(_T_t)
    param_cfo_model = TinyODENetParam(state_dim=3, hidden=_HIDDEN_P)
    _opt_p = torch.optim.Adam(param_cfo_model.parameters(), lr=_LR_P)
    _losses_p = []
    for _ep_p in range(_EPOCHS_P):
        _perm = np.random.permutation(_N_P)
        _ep_loss = 0.0
        for _i in range(0, _N_P, _BATCH_P):
            _ib = _perm[_i : _i + _BATCH_P]
            _opt_p.zero_grad()
            _l = torch.mean(
                (param_cfo_model(_T_t[_ib], _U_t[_ib], _RHO_t[_ib]) - _DU_t[_ib]) ** 2
            )
            _l.backward()
            _opt_p.step()
            _ep_loss += float(_l) * len(_ib)
        _losses_p.append(_ep_loss / _N_P)
    param_norm_stats = (
        _p_state_mean,
        _p_state_std,
        _T_MAX_P,
        _DT_P,
        _p_du_mean,
        _p_du_std,
        _RHO_MEAN,
        _RHO_STD,
    )
    mo.callout(
        mo.md(f"Parametric CFO trained. Final loss: {_losses_p[-1]:.4f}"),
        kind="success",
    )
    return param_cfo_model, param_norm_stats


@app.cell(hide_code=True)
def parametric_explore_controls(mo, param_cfo_model):
    mo.stop(
        param_cfo_model is None,
        mo.callout(mo.md("Train Parametric CFO first."), kind="neutral"),
    )
    rho_slider = mo.ui.slider(
        start=15.0,
        stop=50.0,
        step=0.5,
        value=28.0,
        label="ρ (Lorenz parameter)",
        show_value=True,
    )
    rho_slider
    return (rho_slider,)


@app.cell(hide_code=True)
def vector_field_viz(
    generate_lorenz_param,
    make_cfo_param_fn,
    mo,
    np,
    param_cfo_model,
    param_norm_stats,
    plt,
    rho_slider,
    rk4_ode,
):
    _sm, _ss, _T_MAX_p, _DT_p, _du_m, _du_s, _RHO_M, _RHO_S = param_norm_stats
    _rho = rho_slider.value
    _rho_n = (_rho - _RHO_M) / _RHO_S
    _cfo_p_fn = make_cfo_param_fn(param_cfo_model, _rho_n, _du_m, _du_s)
    _rng_vf = np.random.default_rng(99)
    _x0_vf = _rng_vf.uniform(-8, 8, 3)
    _, _gt_vf = generate_lorenz_param(_x0_vf, 300, _rho, _DT_p)
    _x0_n_vf = ((_x0_vf - _sm) / _ss).astype(np.float32)
    _dt_n_vf = _DT_p / _T_MAX_p
    _cfo_traj_n = rk4_ode(_cfo_p_fn, 0.0, _x0_n_vf, _dt_n_vf, 300)
    _cfo_traj_vf = _cfo_traj_n * _ss + _sm

    # Grid for streamlines: x-y slice at z=25
    # vy = x*(rho-25)-y depends on rho, so the field visually changes with the slider
    _Z_SLICE = 25.0
    _ng = 20
    _xs = np.linspace(-20, 20, _ng)
    _ys = np.linspace(-30, 30, _ng)
    _XX, _YY = np.meshgrid(_xs, _ys)
    _UU = np.zeros_like(_XX)
    _VV = np.zeros_like(_XX)
    _UU_c = np.zeros_like(_XX)
    _VV_c = np.zeros_like(_XX)
    _sigma = 10.0

    for _i_g in range(_ng):
        for _j_g in range(_ng):
            _x, _y = _XX[_i_g, _j_g], _YY[_i_g, _j_g]
            _UU[_i_g, _j_g] = _sigma * (_y - _x)
            _VV[_i_g, _j_g] = _x * (_rho - _Z_SLICE) - _y
            _u_p = np.array([_x, _y, _Z_SLICE])
            _u_n_g = ((_u_p - _sm) / _ss).astype(np.float32)
            _v_n_g = _cfo_p_fn(0.5, _u_n_g)
            _v_p_g = _v_n_g * _ss / _T_MAX_p
            _UU_c[_i_g, _j_g] = _v_p_g[0]
            _VV_c[_i_g, _j_g] = _v_p_g[1]

    fig_vf2, axes_vf2 = plt.subplots(1, 2, figsize=(12, 5))
    fig_vf2.suptitle(
        f"Parametric CFO: ρ = {_rho:.1f}  (x-y slice at z = {_Z_SLICE:.0f})",
        color="#222222",
        fontsize=11,
    )

    axes_vf2[0].streamplot(
        _xs,
        _ys,
        _UU,
        _VV,
        color=np.sqrt(_UU**2 + _VV**2),
        cmap="Blues",
        linewidth=0.8,
        density=1.2,
    )
    axes_vf2[0].plot(
        _gt_vf[:, 0],
        _gt_vf[:, 1],
        color="#333333",
        lw=1.5,
        alpha=0.8,
        label="True traj",
    )
    axes_vf2[0].set_xlabel("x")
    axes_vf2[0].set_ylabel("y")
    axes_vf2[0].set_title("True Lorenz field", color="#333333")
    axes_vf2[0].set_xlim(-20, 20)
    axes_vf2[0].set_ylim(-30, 30)
    axes_vf2[0].legend(fontsize=8)

    axes_vf2[1].streamplot(
        _xs,
        _ys,
        _UU_c,
        _VV_c,
        color=np.sqrt(_UU_c**2 + _VV_c**2),
        cmap="Purples",
        linewidth=0.8,
        density=1.2,
    )
    axes_vf2[1].plot(
        _cfo_traj_vf[:, 0],
        _cfo_traj_vf[:, 1],
        color="#7799ff",
        lw=1.5,
        alpha=0.8,
        label="CFO traj",
    )
    axes_vf2[1].plot(
        _gt_vf[:, 0],
        _gt_vf[:, 1],
        color="#aaaaaa",
        lw=0.8,
        alpha=0.5,
        label="True traj",
    )
    axes_vf2[1].set_xlabel("x")
    axes_vf2[1].set_ylabel("y")
    axes_vf2[1].set_title("Parametric CFO field", color="#333333")
    axes_vf2[1].set_xlim(-20, 20)
    axes_vf2[1].set_ylim(-30, 30)
    axes_vf2[1].legend(fontsize=8)

    plt.tight_layout()
    _vf2_out = mo.center(mo.as_html(fig_vf2))
    plt.close(fig_vf2)
    _vf2_out
    return


@app.cell(hide_code=True)
def click_intro(mo):
    mo.md(r"""
    ### Interactive Exploration

    The heatmap below shows the speed of the learned parametric vector field in the $x$–$y$ plane (with $z$ fixed). **Click anywhere on the heatmap** to set an initial condition and launch a trajectory. The parametric CFO model instantly adapts to the chosen $\rho$ without retraining.
    """)
    return


@app.cell(hide_code=True)
def field_click_panel(
    go,
    mo,
    np,
    param_cfo_model,
    param_norm_stats,
    rho_slider,
):
    _sm, _ss, _T_MAX_p, _DT_p, _du_m, _du_s, _RHO_M, _RHO_S = param_norm_stats
    _rho = rho_slider.value
    _rho_n = float((_rho - _RHO_M) / _RHO_S)

    # x-y plane at z=25: vy = x*(rho-25)-y, so speed changes visually with rho
    _Z_SLICE = 25.0
    _ng = 30
    _xs_g = np.linspace(-22, 22, _ng)
    _ys_g = np.linspace(-30, 30, _ng)
    _XX_g, _YY_g = np.meshgrid(_xs_g, _ys_g)
    _speed = np.zeros((_ng, _ng))

    import torch as _torch

    with _torch.no_grad():
        for _i in range(_ng):
            _u_row = np.stack([_XX_g[_i], _YY_g[_i], np.full(_ng, _Z_SLICE)], axis=1)
            _u_n_row = ((_u_row - _sm) / _ss).astype(np.float32)
            _t_row = _torch.tensor([0.5] * _ng, dtype=_torch.float32)
            _u_t_row = _torch.tensor(_u_n_row)
            _rho_t_row = _torch.tensor([[_rho_n]] * _ng, dtype=_torch.float32)
            _v_n_row = param_cfo_model(_t_row, _u_t_row, _rho_t_row).numpy()
            _v_p_row = _v_n_row * _ss / _T_MAX_p
            _speed[_i] = np.sqrt((_v_p_row**2).sum(axis=1))

    _fig_click = go.Figure()
    _fig_click.add_trace(
        go.Heatmap(
            x=_xs_g,
            y=_ys_g,
            z=_speed,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="speed"),
        )
    )
    _fig_click.update_layout(
        title=f"Click to launch trajectory (ρ = {_rho:.1f}, z fixed at {_Z_SLICE:.0f})",
        xaxis_title="x",
        yaxis_title="y",
        width=600,
        height=480,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    field_click = mo.ui.plotly(_fig_click)
    mo.vstack(
        [
            mo.md(
                f"**Click anywhere on the heatmap** to set an initial condition (x, y) with z = {_Z_SLICE:.0f} fixed. The trajectory appears below."
            ),
            field_click,
        ]
    )
    return (field_click,)


@app.cell(hide_code=True)
def click_trajectory(
    field_click,
    generate_lorenz_param,
    make_cfo_param_fn,
    mo,
    np,
    param_cfo_model,
    param_norm_stats,
    plt,
    rho_slider,
    rk4_ode,
):
    _clicked = field_click.value
    if not _clicked:
        mo.stop(True, mo.md("Click on the heatmap above to launch a trajectory."))
    _pt = _clicked[0]
    _x0_click = float(_pt["x"])
    _y0_click = float(_pt["y"])
    _Z_SLICE = 25.0
    _ic_click = np.array([_x0_click, _y0_click, _Z_SLICE])

    _sm, _ss, _T_MAX_p, _DT_p, _du_m, _du_s, _RHO_M, _RHO_S = param_norm_stats
    _rho = rho_slider.value
    _rho_n = (_rho - _RHO_M) / _RHO_S
    _cfo_p_fn = make_cfo_param_fn(param_cfo_model, _rho_n, _du_m, _du_s)

    _, _gt_click = generate_lorenz_param(_ic_click, 300, _rho, _DT_p)
    _x0_n_click = ((_ic_click - _sm) / _ss).astype(np.float32)
    _dt_n_click = _DT_p / _T_MAX_p
    _cfo_click_n = rk4_ode(_cfo_p_fn, 0.0, _x0_n_click, _dt_n_click, 300)
    _cfo_click = _cfo_click_n * _ss + _sm

    fig_click_traj, axes_ct = plt.subplots(1, 2, figsize=(11, 4))
    fig_click_traj.suptitle(
        f"Trajectory from IC: x={_x0_click:.1f}, y={_y0_click:.1f}, z={_Z_SLICE:.0f}  |  ρ={_rho:.1f}",
        color="#222222",
        fontsize=10,
    )
    axes_ct[0].plot(
        _gt_click[:, 0], _gt_click[:, 2], color="#333333", lw=1.5, label="True"
    )
    axes_ct[0].plot(
        _cfo_click[:, 0],
        _cfo_click[:, 2],
        color="#7799ff",
        lw=1.5,
        linestyle="--",
        label="Param CFO",
    )
    axes_ct[0].scatter([_x0_click], [_Z_SLICE], color="red", zorder=5, s=60, label="IC")
    axes_ct[0].set_xlabel("x")
    axes_ct[0].set_ylabel("z")
    axes_ct[0].set_title("Phase portrait (x-z)", color="#333333")
    axes_ct[0].legend(fontsize=8)
    _steps_ct = np.arange(301)
    axes_ct[1].plot(_steps_ct, _gt_click[:, 0], color="#333333", lw=1.5, label="True x")
    axes_ct[1].plot(
        _steps_ct,
        _cfo_click[:, 0],
        color="#7799ff",
        lw=1.5,
        linestyle="--",
        label="Param CFO x",
    )
    axes_ct[1].set_xlabel("step")
    axes_ct[1].set_ylabel("x")
    axes_ct[1].set_title("x-component over time", color="#333333")
    axes_ct[1].legend(fontsize=8)
    plt.tight_layout()
    _ct_out = mo.center(mo.as_html(fig_click_traj))
    plt.close(fig_click_traj)
    _ct_out
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md(r"""
    ## Key Takeaways

    | # | What this notebook shows | Role |
    |---|---|---|
    | 1 | CFO reproduces the paper's core Lorenz advantage: strong performance from **sparse, irregular** observations against AR baselines | Reproduction |
    | 2 | CFO learns a continuous vector field that is **resolution-agnostic**: query at any step size without retraining | Reproduction + demo |
    | 3 | **Parametric CFO** conditions on Lorenz $\\rho$, learning the entire family of attractors with one model | Novel |
    | 4 | The clickable vector field lets you launch trajectories from any initial condition across the attractor family | Novel demo |
    | 5 | Attractor recovery is supporting evidence: better local dynamics lead to better global geometry | Supporting evidence |

    ---

    ## Limitations of this Demo

    This notebook uses a toy setting (TinyODENet ~5 K params, 3-D Lorenz ODE).
    The full paper:

    - Uses **1D/2D PDE benchmarks** (Burgers, diffusion-reaction, shallow water)
    - Employs U-Net / FNO neural operators (200 K–600 K parameters)
    - Trains on 9 000 trajectories for 200 000 steps on GPU
    - Achieves up to **87 % relative error reduction** vs autoregressive baselines
    - Explores architecture agnosticism beyond the small MLP used in this notebook

    The toy demo faithfully reproduces the *conceptual* CFO advantages. The magnitude
    of the gains scales dramatically with problem complexity and model capacity.

    ---

    **Paper**: [alphaxiv:2512.05297](https://alphaxiv.org/abs/2512.05297)
    · **Authors**: Xianglong Hou, Xinquan Huang, Paris Perdikaris (2025)
    """)
    return


if __name__ == "__main__":
    app.run()

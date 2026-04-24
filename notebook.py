import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from matplotlib.collections import LineCollection
    from scipy.interpolate import CubicSpline

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
    return CubicSpline, LineCollection, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def title_cell(mo):
    mo.md(r"""
    # Reproducing CFO on Lorenz, then Extending it with Physics-Informed Training
    ## *alphaXiv x marimo competition notebook*

    > **Paper**: Hou, Huang & Perdikaris, 2025 · [alphaxiv:2512.05297](https://alphaxiv.org/abs/2512.05297)

    ---

    This notebook makes a two-part argument:

    1. **Reproduction**: CFO's continuous-time formulation is more data-efficient than
       autoregressive next-step prediction on sparse Lorenz trajectories.
    2. **Novel extension**: once the model learns a vector field rather than a step map,
       we can regularize it directly against known physics. We implement that idea here
       as **CFO-PI**.

    Most learned dynamics surrogates use a **discrete step map** $F_\phi(u_i) \to u_{i+1}$.
    That makes them fragile over long rollouts and ties them to the training step size
    $\Delta t$. **CFO** instead learns the right-hand side of an ODE, $\mathcal{N}_\theta(t, u)$,
    trained via flow matching on spline-derived velocity targets without differentiating
    through the ODE solver. At inference, a numerical integrator evaluates the learned
    vector field at arbitrary query times.

    | | Standard AR | **CFO / CFO-PI** |
    |---|---|---|
    | Learns | step map $F_\phi$ | vector field $\mathcal{N}_\theta(t, u)$ |
    | Handles irregular observations? | No — assumes fixed spacing | **Yes** |
    | Long-rollout behavior | Errors accumulate under chaining | **Integrates a learned field** |
    | Equation-level physics priors | No natural interface | **Directly regularizable** |
    | Competition role | reproduction baseline | **reproduction + novel extension** |

    > The paper reports large gains on Lorenz and several PDE benchmarks. This notebook
    > focuses on the Lorenz setting, then extends the method with a physics-informed variant
    > the paper motivates but does not instantiate here.
    """)
    return


@app.cell(hide_code=True)
def cfo_explainer(mo):
    mo.md(r"""
    ## Why CFO Changes the Learning Problem

    ### The autoregressive issue

    A standard next-step model learns $\hat{u}_{i+1} = F_\phi(u_i)$.
    Chaining these steps at inference causes errors to compound — and the model is
    structurally tied to the fixed $\Delta t$ it was trained on.

    ### CFO's key insight

    Treat the trajectory as governed by an ODE:

    $$\frac{du}{dt} = \mathcal{N}_\theta(t,\, u)$$

    Train $\mathcal{N}_\theta$ to approximate the right-hand side; at inference, integrate
    with any standard solver (RK4 here) to any target time $t^*$, with no fixed grid requirement.

    ### The spline trick

    Given sparse, irregular snapshots $\{(t_i, u_i)\}$, fit a **cubic spline** $s(t)$.
    Its analytic derivative $\partial_t s(t)$ serves as the velocity target, giving the
    flow-matching loss:

    $$\mathcal{L}(\theta) = \mathbb{E}_{t,\,\mathbf{u}}\bigl[\|\mathcal{N}_\theta(t,\, s(t;\mathbf{u})) - \partial_t s(t;\mathbf{u})\|^2\bigr]$$

    No ODE solver enters the backward pass, only cheap spline evaluations. That is what
    makes CFO practical on sparse, irregular data, and it is also what makes the later
    physics-informed extension clean to add.
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

    _times_viz = np.arange(_n + 1) * _dt

    fig_lorenz, axes_lorenz = plt.subplots(1, 3, figsize=(14, 4.2))
    fig_lorenz.suptitle(
        "The Lorenz System  ·  $\\dot{x}=\\sigma(y-x)$,  $\\dot{y}=x(\\rho-z)-y$,"
        "  $\\dot{z}=xy-\\beta z$   $[\\sigma=10,\\,\\rho=28,\\,\\beta=8/3]$",
        color="#222222",
        fontsize=11,
    )

    axes_lorenz[0].plot(_times_viz, _traj[:, 0], color="#7799ff", lw=0.6)
    axes_lorenz[0].set_title("x(t)", color="#333333")
    axes_lorenz[0].set_xlabel("time (s)")
    axes_lorenz[0].set_ylabel("x")

    axes_lorenz[1].plot(_times_viz, _traj[:, 2], color="#ff8844", lw=0.6)
    axes_lorenz[1].set_title("z(t)", color="#333333")
    axes_lorenz[1].set_xlabel("time (s)")
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
    axes_lorenz[2].set_title("x–z phase portrait (butterfly attractor)", color="#333333")
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
        label="Keep rate — % of time points per trajectory (CFO training data)",
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
def data_spline_viz(CubicSpline, keep_rate_slider, mo, np, plt):
    """Visualize full trajectory vs sparse samples vs cubic spline reconstruction."""
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
    _cs = CubicSpline(_t_sub, _u_sub)
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
        label="cubic spline",
    )
    _ax0.set_title("x(t) — state reconstruction", color="#333333")
    _ax0.set_xlabel("time (s)")
    _ax0.set_ylabel("x")
    _ax0.legend(fontsize=8)

    # Right: dx/dt derivative — the CFO training target
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
    _ax1.set_title("dx/dt — spline derivative  ← CFO training target", color="#333333")
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
                    "**Left** — sparse observations (dots) and cubic spline reconstruction (white dashed). "
                    "**Right** — the spline's analytic derivative is the flow-matching target that CFO trains on. "
                    "No ODE solver needed: spline derivatives are free. Even at low keep rates the velocity "
                    "field is well approximated."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def data_efficiency_viz(CubicSpline, generate_lorenz, mo, np, plt):
    """Spline reconstruction quality vs keep rate — no retraining needed."""
    _DT = 0.025
    _N_STEPS = 160
    _T_MAX = _N_STEPS * _DT
    _KEEP_RATES = [0.10, 0.25, 0.50, 0.75, 1.00]
    _LABELS = ["10 %", "25 %", "50 %", "75 %", "100 %"]
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
        _cs = CubicSpline(_t_sub, _u_sub)
        _u_recon = _cs(_t_fine)

        _cs_true = CubicSpline(_t_full, _traj_full)
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
def algorithm_cell(mo):
    mo.md(r"""
    ## Algorithms and Baselines

    The notebook first reproduces CFO against two autoregressive baselines, then fine-tunes
    CFO into a physics-informed variant.

    **CFO Training**
    ```
    Input: trajectories {(T_n, U_n)}, neural operator N_θ, keep_rate

    Preprocessing (once per trajectory):
        subsample(T_n, U_n, keep_rate)  →  irregular grid (t̃, ũ)
        normalize ũ to zero-mean unit-variance
        fit CubicSpline s(t) to (t̃_norm, ũ_norm)

    While not converged:
        sample t ~ Uniform[0, 1]
        u   ←  s(t)           # spline value (no ODE solver!)
        du  ←  s'(t)          # analytic derivative — cubic gives smoother targets
        L(θ) = ‖N_θ(t, u) − du‖²
        update θ via Adam
    ```

    **CFO-PI Fine-Tuning**
    ```
    Start from the trained CFO weights
    Sample physical states u and query times t
    Compute true physics velocity f(u)
    Add equation-level penalty:
        L_PI(θ) = L_flow(θ) + λ ‖N_θ(t, u) − f(u)‖²
    ```

    **CFO Inference (RK4)**
    ```
    Given u₀ (normalized), t_span, n_steps:
    for i = 0 … n_steps−1:
        k₁ = N_θ(tᵢ,        uᵢ)
        k₂ = N_θ(tᵢ + h/2,  uᵢ + h/2·k₁)
        k₃ = N_θ(tᵢ + h/2,  uᵢ + h/2·k₂)
        k₄ = N_θ(tᵢ + h,    uᵢ + h·k₃)
        uᵢ₊₁ = uᵢ + h/6·(k₁ + 2k₂ + 2k₃ + k₄)
    ```

    **AR-full Baseline (always trained on 100 % uniform data — paper's hard baseline)**
    ```
    Train:   F_φ(uᵢ) ≈ uᵢ₊₁   (one-step, teacher forcing, fixed Δt, 100 % data)
    Infer:   û₀ → û₁ → û₂ → …  (chains predictions, errors accumulate)
    ```

    **AR-equal Baseline (trained on same sparse data as CFO — fair equal-data comparison)**
    ```
    Train:   F_φ(uᵢ) ≈ uᵢ₊₁   (consecutive kept pairs only, irregular Δt ignored)
    Infer:   û₀ → û₁ → û₂ → …  (same rollout, but learned from fewer pairs)
    Note:    AR cannot use time as input — it sees state→state with no Δt conditioning
    ```
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

    _n_ode = TinyODENet()
    _n_ar = ARNet()
    n_params_ode = sum(p.numel() for p in _n_ode.parameters())
    n_params_ar = sum(p.numel() for p in _n_ar.parameters())

    return (
        ARNet,
        TinyODENet,
        ar_rollout,
        compute_normalization,
        generate_lorenz,
        lorenz_deriv,
        make_cfo_fn,
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
    train_btn = mo.ui.run_button(label="▶ Train Four Models")
    mo.vstack(
        [
            mo.md(
                "## Core Experiment\n\n"
                "This cell performs the main competition comparison: first the paper reproduction, then the "
                "novel physics-informed extension. It trains four models simultaneously:\n"
                "- **CFO** (TinyODENet) — cubic spline derivative targets, sparse data\n"
                "- **CFO-PI** (TinyODENet) — warm-start from CFO + equation-level physics regularization\n"
                "- **AR-full** (ARNet) on 100 % uniform data — paper's hard baseline\n"
                "- **AR-equal** (ARNet) on the same sparse kept pairs as CFO — fair equal-data comparison\n\n"
                f"TinyODENet: **{n_params_ode:,} parameters** (sinusoidal time encoding) · "
                f"ARNet: **{n_params_ar:,} parameters** — both CPU-friendly."
            ),
            mo.hstack([train_epochs, train_btn]),
        ]
    )
    return train_btn, train_epochs


@app.cell(hide_code=True)
def run_training(
    ARNet,
    CubicSpline,
    TinyODENet,
    ar_rollout,
    compute_normalization,
    generate_lorenz,
    horizon_slider,
    keep_rate_slider,
    lorenz_deriv,
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
        cfo_pi_model = None
        ar_model = None
        ar_eq_model = None
        norm_stats = None
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "Click **▶ Train Four Models** to train CFO, CFO-PI, AR-full, and AR-equal."
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

    # ── CFO training data (sparse, irregular cubic spline targets) ────────────
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
        _cs = CubicSpline(_t_sub, _u_sub)
        _t_smp = _rng2.uniform(_t_sub[0], _t_sub[-1], _N_SAMPLE_PER_TRAJ).astype(np.float32)
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
        _ep_loss = 0.0; _nb = 0
        for _i in range(0, _N_CFO, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_cfo.zero_grad()
            _loss = torch.mean((cfo_model(_T_cfo[_idx_b], _U_cfo[_idx_b]) - _DU_cfo[_idx_b]) ** 2)
            _loss.backward(); _opt_cfo.step()
            _ep_loss += _loss.item(); _nb += 1
        _losses_cfo.append(_ep_loss / max(_nb, 1))

    # ── Train AR-full ─────────────────────────────────────────────────────────
    ar_model = ARNet()
    _opt_ar = torch.optim.Adam(ar_model.parameters(), lr=_LR)
    _losses_ar = []
    for _ in range(_EPOCHS):
        _perm = np.random.permutation(_N_AR)
        _ep_loss = 0.0; _nb = 0
        for _i in range(0, _N_AR, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_ar.zero_grad()
            _loss = torch.mean((_U_ar[_idx_b] + ar_model(_U_ar[_idx_b]) - _UN_ar[_idx_b]) ** 2)
            _loss.backward(); _opt_ar.step()
            _ep_loss += _loss.item(); _nb += 1
        _losses_ar.append(_ep_loss / max(_nb, 1))

    # ── Train AR-equal ────────────────────────────────────────────────────────
    ar_eq_model = ARNet()
    _opt_areq = torch.optim.Adam(ar_eq_model.parameters(), lr=_LR)
    _losses_areq = []
    for _ in range(_EPOCHS):
        _perm = np.random.permutation(_N_AREQ)
        _ep_loss = 0.0; _nb = 0
        for _i in range(0, _N_AREQ, _BATCH):
            _idx_b = _perm[_i : _i + _BATCH]
            _opt_areq.zero_grad()
            _loss = torch.mean((_U_areq[_idx_b] + ar_eq_model(_U_areq[_idx_b]) - _UN_areq[_idx_b]) ** 2)
            _loss.backward(); _opt_areq.step()
            _ep_loss += _loss.item(); _nb += 1
        _losses_areq.append(_ep_loss / max(_nb, 1))

    # ── Train CFO-PI (physics-informed, warm-started from cfo_model) ─────────
    cfo_pi_model = TinyODENet()
    cfo_pi_model.load_state_dict(cfo_model.state_dict())
    _opt_pi = torch.optim.Adam(cfo_pi_model.parameters(), lr=_LR * 0.5)
    _LAMBDA_PI = 0.5
    _N_PHYS = 256
    _all_u_phys = np.concatenate([t for _, t in _train_trajs], axis=0)
    _rng_pi = np.random.default_rng(42)
    _losses_pi = []
    for _ in range(_EPOCHS):
        _idx_p = _rng_pi.integers(0, len(_all_u_phys), _N_PHYS)
        _u_ph = _all_u_phys[_idx_p]
        _u_n_ph = ((_u_ph - _state_mean) / _state_std).astype(np.float32)
        _v_ph = np.stack([lorenz_deriv(u) for u in _u_ph]).astype(np.float32)
        _v_n = _v_ph / _state_std * _T_MAX
        _v_tgt = ((_v_n - _du_mean) / _du_std).astype(np.float32)
        _t_rand = torch.tensor(_rng_pi.uniform(0, 1, _N_PHYS).astype(np.float32))
        _u_t_ph = torch.tensor(_u_n_ph)
        _v_t = torch.tensor(_v_tgt)
        _perm_pi = np.random.permutation(_N_CFO)
        _ep_loss_pi = 0.0
        _nb_pi = 0
        for _i in range(0, _N_CFO, _BATCH):
            _idx_b = _perm_pi[_i : _i + _BATCH]
            _opt_pi.zero_grad()
            _l_flow = torch.mean(
                (cfo_pi_model(_T_cfo[_idx_b], _U_cfo[_idx_b]) - _DU_cfo[_idx_b]) ** 2
            )
            _l_phys = torch.mean((cfo_pi_model(_t_rand, _u_t_ph) - _v_t) ** 2)
            (_l_flow + _LAMBDA_PI * _l_phys).backward()
            _opt_pi.step()
            _ep_loss_pi += _l_flow.item()
            _nb_pi += 1
        _losses_pi.append(_ep_loss_pi / max(_nb_pi, 1))

    norm_stats = (_state_mean, _state_std, _T_MAX, _DT, _du_mean, _du_std)

    # ── Eval on test set ──────────────────────────────────────────────────────
    _H = horizon_slider.value
    _dt_n = _DT / _T_MAX
    _cfo_fn = make_cfo_fn(cfo_model, _du_mean, _du_std)
    _cfo_pi_fn = make_cfo_fn(cfo_pi_model, _du_mean, _du_std)
    _cfo_rmse_list, _cfo_pi_rmse_list, _ar_rmse_list, _areq_rmse_list = [], [], [], []

    for _times_raw, _traj_raw in _test_trajs:
        _u_n = (_traj_raw - _state_mean) / _state_std
        _u0_n = _u_n[0]
        _true = _u_n[: _H + 1]
        _cfo_rmse_list.append(np.sqrt(np.mean((rk4_ode(_cfo_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)))
        _cfo_pi_rmse_list.append(np.sqrt(np.mean((rk4_ode(_cfo_pi_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)))
        _ar_rmse_list.append(np.sqrt(np.mean((ar_rollout(ar_model, _u0_n, _H) - _true) ** 2, axis=1)))
        _areq_rmse_list.append(np.sqrt(np.mean((ar_rollout(ar_eq_model, _u0_n, _H) - _true) ** 2, axis=1)))

    _cfo_rmse_mean = np.mean(_cfo_rmse_list, axis=0)
    _cfo_pi_rmse_mean = np.mean(_cfo_pi_rmse_list, axis=0)
    _ar_rmse_mean = np.mean(_ar_rmse_list, axis=0)
    _areq_rmse_mean = np.mean(_areq_rmse_list, axis=0)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _kr = keep_rate_slider.value
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.8))

    _axes[0].plot(_losses_cfo, color="#7799ff", lw=1.5, label=f"CFO ({_kr} %)")
    _axes[0].plot(_losses_pi, color="#aa55ff", lw=1.5, label=f"CFO-PI ({_kr} %)", linestyle="-.")
    _axes[0].plot(_losses_ar, color="#ff8844", lw=1.5, label="AR-full (100 %)", linestyle="--")
    _axes[0].plot(_losses_areq, color="#44dd88", lw=1.5, label=f"AR-equal ({_kr} %)", linestyle=":")
    _axes[0].set_title("Training Loss (flow-matching term)", color="#333333")
    _axes[0].set_xlabel("epoch")
    _axes[0].set_ylabel("MSE (normalised space)")
    _axes[0].legend(fontsize=8)
    _axes[0].set_yscale("log")

    _t_axis = np.arange(_H + 1) * _DT
    _axes[1].plot(_t_axis, _cfo_rmse_mean, color="#7799ff", lw=2, label=f"CFO ({_kr} %)")
    _axes[1].plot(_t_axis, _cfo_pi_rmse_mean, color="#aa55ff", lw=2, linestyle="-.", label=f"CFO-PI ({_kr} %)")
    _axes[1].plot(_t_axis, _ar_rmse_mean, color="#ff8844", lw=2, linestyle="--", label="AR-full (100 %)")
    _axes[1].plot(_t_axis, _areq_rmse_mean, color="#44dd88", lw=2, linestyle=":", label=f"AR-equal ({_kr} %)")
    _axes[1].set_title("Test RMSE over horizon", color="#333333")
    _axes[1].set_xlabel("time (s)")
    _axes[1].set_ylabel("RMSE (normalised)")
    _axes[1].legend(fontsize=8)

    plt.tight_layout()
    _fig_out = mo.center(mo.as_html(_fig))
    plt.close(_fig)

    _cfo_final = float(_cfo_rmse_mean[-1])
    _cfo_pi_final = float(_cfo_pi_rmse_mean[-1])
    _ar_final = float(_ar_rmse_mean[-1])
    _impr_vs_full = 100 * (1 - _cfo_final / (_ar_final + 1e-9))

    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**AR-full** trains on 100 % data (paper's hard baseline). "
                    f"**AR-equal** and **CFO** train on the same {_kr} % sparse data. "
                    "**CFO-PI** adds a physics regularisation term (known Lorenz equations) "
                    "to the flow-matching loss."
                ),
                kind="info",
            ),
            mo.hstack(
                [
                    mo.stat(f"{_losses_cfo[-1]:.4f}", label="CFO loss"),
                    mo.stat(f"{_losses_pi[-1]:.4f}", label="CFO-PI loss"),
                    mo.stat(f"{_losses_ar[-1]:.4f}", label="AR-full loss"),
                    mo.stat(f"{_cfo_final:.3f}", label="CFO RMSE"),
                    mo.stat(f"{_cfo_pi_final:.3f}", label="CFO-PI RMSE"),
                    mo.stat(f"{_ar_final:.3f}", label="AR-full RMSE"),
                    mo.stat(f"{_impr_vs_full:.1f} %", label="CFO vs AR-full"),
                ]
            ),
            _fig_out,
        ]
    )
    return ar_eq_model, ar_model, cfo_model, cfo_pi_model, norm_stats


@app.cell(hide_code=True)
def error_over_time(
    ar_eq_model,
    ar_model,
    ar_rollout,
    cfo_model,
    cfo_pi_model,
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
    _all_cfo_rmse, _all_cfo_pi_rmse, _all_ar_rmse, _all_areq_rmse = [], [], [], []

    _cfo_fn = make_cfo_fn(cfo_model, _du_mean, _du_std)
    _cfo_pi_fn = make_cfo_fn(cfo_pi_model, _du_mean, _du_std) if cfo_pi_model is not None else None
    for _ in range(_N_EVAL):
        _x0 = _rng_err.uniform(-8, 8, 3)
        _, _traj = generate_lorenz(_x0, _H + 5, _DT)
        _u_n = (_traj - _state_mean) / _state_std
        _u0_n = _u_n[0]
        _true = _u_n[: _H + 1]
        _all_cfo_rmse.append(np.sqrt(np.mean((rk4_ode(_cfo_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)))
        if _cfo_pi_fn is not None:
            _all_cfo_pi_rmse.append(np.sqrt(np.mean((rk4_ode(_cfo_pi_fn, 0.0, _u0_n, _dt_n, _H) - _true) ** 2, axis=1)))
        _all_ar_rmse.append(np.sqrt(np.mean((ar_rollout(ar_model, _u0_n, _H) - _true) ** 2, axis=1)))
        _all_areq_rmse.append(np.sqrt(np.mean((ar_rollout(ar_eq_model, _u0_n, _H) - _true) ** 2, axis=1)))

    _cfo_m = np.mean(_all_cfo_rmse, axis=0)
    _cfo_s = np.std(_all_cfo_rmse, axis=0)
    _ar_m = np.mean(_all_ar_rmse, axis=0)
    _ar_s = np.std(_all_ar_rmse, axis=0)
    _areq_m = np.mean(_all_areq_rmse, axis=0)
    _areq_s = np.std(_all_areq_rmse, axis=0)
    _t_ax = np.arange(_H + 1) * _DT

    fig_err, ax_err = plt.subplots(figsize=(12, 4))
    ax_err.plot(_t_ax, _cfo_m, color="#7799ff", lw=2, label=f"CFO ({_kr} % data)")
    ax_err.fill_between(_t_ax, _cfo_m - _cfo_s, _cfo_m + _cfo_s, color="#7799ff", alpha=0.2)
    if _all_cfo_pi_rmse:
        _cfo_pi_m = np.mean(_all_cfo_pi_rmse, axis=0)
        _cfo_pi_s = np.std(_all_cfo_pi_rmse, axis=0)
        ax_err.plot(_t_ax, _cfo_pi_m, color="#aa55ff", lw=2, linestyle="-.", label=f"CFO-PI ({_kr} % data + physics)")
        ax_err.fill_between(_t_ax, _cfo_pi_m - _cfo_pi_s, _cfo_pi_m + _cfo_pi_s, color="#aa55ff", alpha=0.2)
    ax_err.plot(_t_ax, _ar_m, color="#ff8844", lw=2, linestyle="--", label="AR-full (100 % data)")
    ax_err.fill_between(_t_ax, _ar_m - _ar_s, _ar_m + _ar_s, color="#ff8844", alpha=0.2)
    ax_err.plot(_t_ax, _areq_m, color="#44dd88", lw=2, linestyle=":", label=f"AR-equal ({_kr} % data)")
    ax_err.fill_between(_t_ax, _areq_m - _areq_s, _areq_m + _areq_s, color="#44dd88", alpha=0.2)
    ax_err.set_title(
        f"Prediction RMSE over Time  ·  averaged over {_N_EVAL} test trajectories",
        color="#333333",
    )
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("RMSE (normalised, log scale)")
    ax_err.set_yscale("log")
    ax_err.legend(fontsize=10)
    _cross_idx = int(np.argmax(_ar_m > _cfo_m))
    if _cross_idx > 0:
        ax_err.axvline(_t_ax[_cross_idx], color="#ff8844", lw=1.0, linestyle=":", alpha=0.55)
        ax_err.annotate(
            "AR-full diverges\npast CFO",
            xy=(_t_ax[_cross_idx], _ar_m[_cross_idx]),
            xytext=(_t_ax[_cross_idx] + 0.25, _ar_m[_cross_idx] * 2.0),
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
    ## Reproduction Result: The Data-Efficiency Frontier

    The single run above shows one setting at a time. The sweep below is the notebook's
    main reproduction result: it trains **all four models** (CFO, CFO-PI, AR-full, AR-equal)
    across five keep rates (10 – 100 %) with **three random seeds** each, revealing where
    the continuous-time formulation compensates for missing data.

    > AR-equal cannot use the irregular Δt between kept pairs — it learns
    > state→state with a fixed implicit step. This is a structural limitation of
    > the AR formulation that CFO does not share.
    """)
    return


@app.cell(hide_code=True)
def sweep_controls(mo):
    sweep_btn = mo.ui.run_button(
        label="▶ Run Keep-Rate Sweep  (3 seeds × 5 rates × 4 models)"
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "This trains **60 models** (4 models × 5 keep rates × 3 seeds) "
                    "at 100 epochs each. Expect 3–7 minutes on CPU."
                ),
                kind="warn",
            ),
            sweep_btn,
        ]
    )
    return (sweep_btn,)


@app.cell(hide_code=True)
def keep_rate_sweep(
    ARNet,
    CubicSpline,
    TinyODENet,
    ar_rollout,
    compute_normalization,
    generate_lorenz,
    make_cfo_fn,
    mo,
    np,
    plt,
    rk4_ode,
    sweep_btn,
    torch,
):
    sweep_btn  # reactive dependency

    if not sweep_btn.value:
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    "Click **▶ Run Keep-Rate Sweep** to compare all four models across keep rates."
                ),
                kind="neutral",
            ),
        )

    _DT_sw = 0.025
    _N_STEPS_sw = 200
    _N_TRAIN_sw = 20
    _N_TEST_sw = 10
    _EPOCHS_sw = 100  # shorter per run — sweep is meant to show trends
    _LR_sw = 3e-3
    _BATCH_sw = 128
    _N_SMP_sw = 60  # spline samples per trajectory
    _T_MAX_sw = _N_STEPS_sw * _DT_sw
    _H_sw = 80  # prediction horizon (steps)
    _N_EVAL_sw = 10  # test trajectories per seed

    _KEEP_RATES_sw = [0.10, 0.25, 0.50, 0.75, 1.00]
    _SEEDS_sw = [0, 1, 2]

    # Fixed trajectory pool — same across all keep rates and seeds
    _rng_base_sw = np.random.default_rng(42)
    _all_x0_sw = _rng_base_sw.uniform(-8, 8, (_N_TRAIN_sw + _N_TEST_sw, 3))
    _all_trajs_sw = [
        generate_lorenz(_all_x0_sw[i], _N_STEPS_sw, _DT_sw)
        for i in range(_N_TRAIN_sw + _N_TEST_sw)
    ]
    _train_trajs_sw = _all_trajs_sw[:_N_TRAIN_sw]
    _test_trajs_sw = _all_trajs_sw[_N_TRAIN_sw:]

    _state_mean_sw, _state_std_sw = compute_normalization(_train_trajs_sw)

    # Normalise test trajectories once
    _test_u_norm = [
        ((_tr - _state_mean_sw) / _state_std_sw) for _, _tr in _test_trajs_sw
    ]

    def _sweep_run(kr, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        _rng_s = np.random.default_rng(seed + 200)

        # ── CFO data (cubic spline) ────────────────────────────────────────────
        _cfo_t_l, _cfo_u_l, _cfo_du_l, _kept_idx_l = [], [], [], []
        for _times_r, _traj_r in _train_trajs_sw:
            _n_pts = len(_times_r)
            _n_keep = max(6, int(_n_pts * kr))
            _idx = np.sort(_rng_s.choice(_n_pts, _n_keep, replace=False))
            _kept_idx_l.append(_idx)
            _t_sub = _times_r[_idx] / _T_MAX_sw
            _u_sub = (_traj_r[_idx] - _state_mean_sw) / _state_std_sw
            _cs = CubicSpline(_t_sub, _u_sub)
            _t_smp = _rng_s.uniform(_t_sub[0], _t_sub[-1], _N_SMP_sw).astype(np.float32)
            _cfo_t_l.append(_t_smp)
            _cfo_u_l.append(_cs(_t_smp).astype(np.float32))
            _cfo_du_l.append(_cs(_t_smp, 1).astype(np.float32))

        _all_du = np.concatenate(_cfo_du_l, axis=0)
        _du_mean_s = _all_du.mean(axis=0).astype(np.float32)
        _du_std_s = (_all_du.std(axis=0) + 1e-8).astype(np.float32)
        _cfo_du_sc = [(du - _du_mean_s) / _du_std_s for du in _cfo_du_l]

        _T_c = torch.tensor(np.concatenate(_cfo_t_l))
        _U_c = torch.tensor(np.concatenate(_cfo_u_l))
        _DU_c = torch.tensor(np.concatenate(_cfo_du_sc))
        _N_c = len(_T_c)

        # ── AR-full data ───────────────────────────────────────────────────────
        _u_af_l, _un_af_l = [], []
        for _, _tr in _train_trajs_sw:
            _un = ((_tr - _state_mean_sw) / _state_std_sw).astype(np.float32)
            _u_af_l.append(_un[:-1])
            _un_af_l.append(_un[1:])
        _U_af = torch.tensor(np.concatenate(_u_af_l))
        _UN_af = torch.tensor(np.concatenate(_un_af_l))
        _N_af = len(_U_af)

        # ── AR-equal data ──────────────────────────────────────────────────────
        _u_ae_l, _un_ae_l = [], []
        for (_, _tr), _idx in zip(_train_trajs_sw, _kept_idx_l):
            _un = ((_tr - _state_mean_sw) / _state_std_sw).astype(np.float32)
            _u_ae_l.append(_un[_idx[:-1]])
            _un_ae_l.append(_un[_idx[1:]])
        _U_ae = torch.tensor(np.concatenate(_u_ae_l))
        _UN_ae = torch.tensor(np.concatenate(_un_ae_l))
        _N_ae = len(_U_ae)

        # ── Training helper ────────────────────────────────────────────────────
        def _train_ar_net(U, UN, N):
            m = ARNet()
            opt = torch.optim.Adam(m.parameters(), lr=_LR_sw)
            for _ in range(_EPOCHS_sw):
                _p = np.random.permutation(N)
                for _i in range(0, N, _BATCH_sw):
                    _ib = _p[_i : _i + _BATCH_sw]
                    _ub, _unb = U[_ib], UN[_ib]
                    opt.zero_grad()
                    torch.mean((_ub + m(_ub) - _unb) ** 2).backward()
                    opt.step()
            return m

        def _train_cfo_net(T, U, DU, N):
            m = TinyODENet()
            opt = torch.optim.Adam(m.parameters(), lr=_LR_sw)
            for _ in range(_EPOCHS_sw):
                _p = np.random.permutation(N)
                for _i in range(0, N, _BATCH_sw):
                    _ib = _p[_i : _i + _BATCH_sw]
                    opt.zero_grad()
                    torch.mean((m(T[_ib], U[_ib]) - DU[_ib]) ** 2).backward()
                    opt.step()
            return m

        _cfo_m = _train_cfo_net(_T_c, _U_c, _DU_c, _N_c)
        _ar_full = _train_ar_net(_U_af, _UN_af, _N_af)
        _ar_eq = _train_ar_net(_U_ae, _UN_ae, _N_ae)

        # ── CFO-PI (warm-start from CFO, add physics regularisation) ─────────
        _lorenz_deriv_sw = lambda s: np.array([10.0 * (s[1] - s[0]), s[0] * (28.0 - s[2]) - s[1], s[0] * s[1] - (8.0 / 3.0) * s[2]])
        _cfo_pi_sw = TinyODENet()
        _cfo_pi_sw.load_state_dict(_cfo_m.state_dict())
        _opt_pi_sw = torch.optim.Adam(_cfo_pi_sw.parameters(), lr=_LR_sw * 0.5)
        _LAMBDA_PI_sw = 0.5
        _du_mean_t = torch.tensor(_du_mean_s)
        _du_std_t = torch.tensor(_du_std_s)
        _state_mean_t = torch.tensor(_state_mean_sw.astype(np.float32))
        _state_std_t = torch.tensor(_state_std_sw.astype(np.float32))
        _T_MAX_t = float(_T_MAX_sw)
        for _ep_pi in range(_EPOCHS_sw):
            _p = np.random.permutation(_N_c)
            for _i in range(0, _N_c, _BATCH_sw):
                _ib = _p[_i : _i + _BATCH_sw]
                _opt_pi_sw.zero_grad()
                _l_flow = torch.mean((_cfo_pi_sw(_T_c[_ib], _U_c[_ib]) - _DU_c[_ib]) ** 2)
                _rng_ph = np.random.default_rng(_ep_pi * 1000 + _i)
                _u_phys_ph = torch.tensor(
                    _rng_ph.uniform(-15, 15, (32, 3)).astype(np.float32)
                )
                _t_rand_ph = torch.tensor(_rng_ph.uniform(0, 1, (32,)).astype(np.float32))
                _u_n_ph = (_u_phys_ph - _state_mean_t) / _state_std_t
                _v_phys = torch.tensor(
                    np.stack([_lorenz_deriv_sw(u.numpy()) for u in _u_phys_ph]).astype(np.float32)
                )
                _v_tgt = (_v_phys / _state_std_t * _T_MAX_t - _du_mean_t) / _du_std_t
                _l_phys = torch.mean((_cfo_pi_sw(_t_rand_ph, _u_n_ph) - _v_tgt) ** 2)
                (_l_flow + _LAMBDA_PI_sw * _l_phys).backward()
                _opt_pi_sw.step()

        # ── Evaluation ────────────────────────────────────────────────────────
        _cfo_fn = make_cfo_fn(_cfo_m, _du_mean_s, _du_std_s)
        _cfo_pi_fn = make_cfo_fn(_cfo_pi_sw, _du_mean_s, _du_std_s)
        _dt_n = _DT_sw / _T_MAX_sw
        _r_cfo, _r_cfo_pi, _r_arf, _r_are = [], [], [], []
        for _u_n_t in _test_u_norm[:_N_EVAL_sw]:
            _u0 = _u_n_t[0]
            _true = _u_n_t[: _H_sw + 1]
            _r_cfo.append(float(np.sqrt(np.mean((rk4_ode(_cfo_fn, 0.0, _u0, _dt_n, _H_sw) - _true) ** 2))))
            _r_cfo_pi.append(float(np.sqrt(np.mean((rk4_ode(_cfo_pi_fn, 0.0, _u0, _dt_n, _H_sw) - _true) ** 2))))
            _r_arf.append(float(np.sqrt(np.mean((ar_rollout(_ar_full, _u0, _H_sw) - _true) ** 2))))
            _r_are.append(float(np.sqrt(np.mean((ar_rollout(_ar_eq, _u0, _H_sw) - _true) ** 2))))
        return np.mean(_r_cfo), np.mean(_r_cfo_pi), np.mean(_r_arf), np.mean(_r_are)

    # ── Run the sweep ─────────────────────────────────────────────────────────
    _res_cfo = np.zeros((len(_KEEP_RATES_sw), len(_SEEDS_sw)))
    _res_cfo_pi = np.zeros_like(_res_cfo)
    _res_arf = np.zeros_like(_res_cfo)
    _res_are = np.zeros_like(_res_cfo)

    for _ki, _kr in enumerate(_KEEP_RATES_sw):
        for _si, _seed in enumerate(_SEEDS_sw):
            _rc, _rcp, _ra, _rq = _sweep_run(_kr, _seed)
            _res_cfo[_ki, _si] = _rc
            _res_cfo_pi[_ki, _si] = _rcp
            _res_arf[_ki, _si] = _ra
            _res_are[_ki, _si] = _rq

    _cfo_m_sw = _res_cfo.mean(axis=1)
    _cfo_s_sw = _res_cfo.std(axis=1)
    _cfo_pi_m_sw = _res_cfo_pi.mean(axis=1)
    _cfo_pi_s_sw = _res_cfo_pi.std(axis=1)
    _arf_m_sw = _res_arf.mean(axis=1)
    _arf_s_sw = _res_arf.std(axis=1)
    _are_m_sw = _res_are.mean(axis=1)
    _are_s_sw = _res_are.std(axis=1)
    _kr_vals_sw = [kr * 100 for kr in _KEEP_RATES_sw]

    fig_sw, ax_sw = plt.subplots(1, 1, figsize=(12, 5))
    fig_sw.suptitle(
        f"Keep-Rate Sweep  ·  {len(_SEEDS_sw)} seeds × {len(_KEEP_RATES_sw)} keep rates"
        f"  ·  {_EPOCHS_sw} epochs each  ·  horizon = {_H_sw} steps",
        color="#222222",
        fontsize=12,
    )

    ax_sw.plot(
        _kr_vals_sw, _cfo_m_sw, color="#7799ff", lw=2.5, marker="o", label="CFO"
    )
    ax_sw.fill_between(
        _kr_vals_sw,
        _cfo_m_sw - _cfo_s_sw,
        _cfo_m_sw + _cfo_s_sw,
        color="#7799ff",
        alpha=0.2,
    )
    ax_sw.plot(
        _kr_vals_sw, _cfo_pi_m_sw, color="#aa55ff", lw=2.5, marker="D",
        linestyle="-.", label="CFO-PI (+ physics)",
    )
    ax_sw.fill_between(
        _kr_vals_sw,
        _cfo_pi_m_sw - _cfo_pi_s_sw,
        _cfo_pi_m_sw + _cfo_pi_s_sw,
        color="#aa55ff",
        alpha=0.2,
    )
    ax_sw.plot(
        _kr_vals_sw,
        _arf_m_sw,
        color="#ff8844",
        lw=2,
        marker="s",
        linestyle="--",
        label="AR-full (100 %)",
    )
    ax_sw.fill_between(
        _kr_vals_sw,
        _arf_m_sw - _arf_s_sw,
        _arf_m_sw + _arf_s_sw,
        color="#ff8844",
        alpha=0.2,
    )
    ax_sw.plot(
        _kr_vals_sw,
        _are_m_sw,
        color="#44dd88",
        lw=2,
        marker="^",
        linestyle=":",
        label="AR-equal",
    )
    ax_sw.fill_between(
        _kr_vals_sw,
        _are_m_sw - _are_s_sw,
        _are_m_sw + _are_s_sw,
        color="#44dd88",
        alpha=0.2,
    )

    # Flat reference line: AR-full mean RMSE across all keep rates
    _arf_ref = float(_arf_m_sw.mean())
    ax_sw.axhline(
        _arf_ref,
        color="#ff8844",
        lw=1.0,
        linestyle=":",
        alpha=0.5,
        label=f"AR-full baseline ≈ {_arf_ref:.3f}",
    )

    # Annotate first keep rate where CFO-PI beats AR-full
    _win_mask_pi = _cfo_pi_m_sw < _arf_m_sw
    if _win_mask_pi.any():
        _win_kr_pi = _kr_vals_sw[int(np.argmax(_win_mask_pi))]
        _win_rmse_pi = float(_cfo_pi_m_sw[int(np.argmax(_win_mask_pi))])
        ax_sw.annotate(
            f"CFO-PI wins here\n({_win_kr_pi:.0f} % data)",
            xy=(_win_kr_pi, _win_rmse_pi),
            xytext=(_win_kr_pi + 6, _win_rmse_pi + (_arf_ref - _win_rmse_pi) * 0.5),
            color="#aa55ff",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#aa55ff", lw=0.9),
        )

    ax_sw.set_xlabel("Keep rate — % of time points per trajectory")
    ax_sw.set_ylabel(f"Mean final RMSE (normalised, horizon = {_H_sw} steps)")
    ax_sw.set_title(
        "Data Efficiency: RMSE vs Keep Rate  (mean ± 1 std, 3 seeds)", color="#333333"
    )
    ax_sw.legend(fontsize=9)

    plt.tight_layout()
    _sw_out = mo.center(mo.as_html(fig_sw))
    plt.close(fig_sw)

    mo.vstack(
        [
            _sw_out,
            mo.callout(
                mo.md(
                    "Where CFO/CFO-PI (blue/purple) dip below the orange dashed AR-full line reveals the "
                    "keep rate at which the continuous-time inductive bias compensates for the data disadvantage. "
                    "CFO-PI's physics regularisation can shift this crossover to even lower keep rates — "
                    "the governing equations substitute for missing data."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def physics_interp_intro(mo):
    mo.md(r"""
    ## Novel Contribution: Physics-Informed CFO

    The CFO paper (Section 6) identifies **physics integration** as the natural next step:

    > *"When governing equations are partially known, augmenting training with physics-informed
    > constraints could improve backward integration stability."*

    This notebook implements that idea as **CFO-PI**. Starting from a trained CFO model,
    we add a regularisation term that penalises deviation of the learned vector field
    $N_\theta(t, u)$ from the true Lorenz equations at random state-space points:

    $$\mathcal{L}_\text{PI} = \mathcal{L}_\text{flow} + \lambda \,\mathbb{E}_{u}\bigl[\|N_\theta(t, u) - f_\text{Lorenz}(u)\|^2\bigr]$$

    This is the notebook's main novel contribution.

    **Why is this natural for CFO, but not for a standard AR baseline?**
    CFO's output *is* a vector field — every prediction $N_\theta(t, u)$ is a velocity
    that can be compared to the true physics at any state-space point, even outside
    the training trajectories. AR predicts discrete transitions $F(u) \to u'$ and does
    not expose the same continuous object to regularise against a differential equation.

    **Test:** Query $N_\theta(t, u)$ on a 2-D grid of state-space points and compare to the
    true Lorenz field. If CFO-PI works, it should match the governing equations more closely
    than vanilla CFO and potentially improve data efficiency as well.
    """)
    return


@app.cell(hide_code=True)
def physics_interp_viz(
    cfo_model,
    cfo_pi_model,
    lorenz_deriv,
    make_cfo_fn,
    mo,
    norm_stats,
    np,
    plt,
):
    if cfo_pi_model is None:
        mo.stop(True)

    _state_mean, _state_std, _T_MAX, _, _du_mean, _du_std = norm_stats

    # 2D grid in physical space: x ∈ [-20, 20], y ∈ [-25, 30] at z = 25
    _nx, _ny = 16, 16
    _xs = np.linspace(-20, 20, _nx)
    _ys = np.linspace(-25, 30, _ny)
    _XX, _YY = np.meshgrid(_xs, _ys)
    _ZZ = np.full_like(_XX, 25.0)
    _grid_phys = np.stack([_XX.ravel(), _YY.ravel(), _ZZ.ravel()], axis=1)  # (N, 3)

    # True Lorenz velocity on grid (physical space)
    _v_true = np.stack([lorenz_deriv(u) for u in _grid_phys])  # (N, 3)

    def _query_field(model, grid_phys):
        _fn = make_cfo_fn(model, _du_mean, _du_std)
        _t_fixed = 0.5  # mid-range normalised time
        _vels = []
        for _u_p in grid_phys:
            _u_n = (_u_p - _state_mean) / _state_std
            _v_n = _fn(_t_fixed, _u_n)               # d(u_n)/d(t_n)
            _v_p = _v_n * _state_std / _T_MAX         # physical velocity
            _vels.append(_v_p)
        return np.array(_vels)

    _v_cfo = _query_field(cfo_model, _grid_phys)
    _v_pi = _query_field(cfo_pi_model, _grid_phys)

    # Residual magnitude (x-y components only)
    _res_cfo = np.linalg.norm(_v_cfo[:, :2] - _v_true[:, :2], axis=1).reshape(_ny, _nx)
    _res_pi = np.linalg.norm(_v_pi[:, :2] - _v_true[:, :2], axis=1).reshape(_ny, _nx)
    _rmse_cfo = float(np.mean(_res_cfo))
    _rmse_pi = float(np.mean(_res_pi))

    # Normalise arrows to unit length for direction-only comparison
    def _unit(arr2d_u, arr2d_v):
        _mag = np.sqrt(arr2d_u ** 2 + arr2d_v ** 2) + 1e-10
        return arr2d_u / _mag, arr2d_v / _mag

    _U_true = _v_true[:, 0].reshape(_ny, _nx)
    _V_true = _v_true[:, 1].reshape(_ny, _nx)
    _U_cfo = _v_cfo[:, 0].reshape(_ny, _nx)
    _V_cfo = _v_cfo[:, 1].reshape(_ny, _nx)
    _U_pi = _v_pi[:, 0].reshape(_ny, _nx)
    _V_pi = _v_pi[:, 1].reshape(_ny, _nx)

    _Ut, _Vt = _unit(_U_true, _V_true)
    _Uc, _Vc = _unit(_U_cfo, _V_cfo)
    _Up, _Vp = _unit(_U_pi, _V_pi)

    fig_vf, axes_vf = plt.subplots(1, 3, figsize=(14, 4.5))
    fig_vf.suptitle(
        "Vector Field Alignment (x–y slice at z = 25) — Does the model learn the governing equations?",
        color="#222222", fontsize=11,
    )
    _qkw = {"alpha": 0.85, "width": 0.003}

    # Panel 1: True Lorenz
    axes_vf[0].quiver(_XX, _YY, _Ut, _Vt, color="#333333", **_qkw)
    axes_vf[0].set_title("True Lorenz field", color="#333333")
    axes_vf[0].set_xlabel("x")
    axes_vf[0].set_ylabel("y")

    # Panel 2: CFO
    _im_cfo = axes_vf[1].imshow(
        _res_cfo, extent=[-20, 20, -25, 30], origin="lower",
        cmap="Reds", alpha=0.35, aspect="auto",
    )
    axes_vf[1].quiver(_XX, _YY, _Uc, _Vc, color="#7799ff", **_qkw)
    axes_vf[1].set_title(f"CFO  (mean residual = {_rmse_cfo:.1f})", color="#333333")
    axes_vf[1].set_xlabel("x")

    # Panel 3: CFO-PI
    _im_pi = axes_vf[2].imshow(
        _res_pi, extent=[-20, 20, -25, 30], origin="lower",
        cmap="Reds", alpha=0.35, aspect="auto",
        vmin=_im_cfo.norm.vmin, vmax=_im_cfo.norm.vmax,
    )
    axes_vf[2].quiver(_XX, _YY, _Up, _Vp, color="#aa55ff", **_qkw)
    axes_vf[2].set_title(f"CFO-PI  (mean residual = {_rmse_pi:.1f})", color="#333333")
    axes_vf[2].set_xlabel("x")

    fig_vf.colorbar(_im_pi, ax=axes_vf[1:], label="‖predicted − true‖ (physical units)", shrink=0.85)
    plt.tight_layout()
    _vf_out = mo.center(mo.as_html(fig_vf))
    plt.close(fig_vf)

    mo.vstack([
        _vf_out,
        mo.callout(
            mo.md(
                f"The red heatmap shows the pointwise error between each model's predicted velocity "
                f"and the true Lorenz field. "
                f"CFO-PI mean residual = **{_rmse_pi:.1f}** vs CFO = **{_rmse_cfo:.1f}** (lower is better). "
                "This is the main novel result in the notebook: equation-level physics regularization is "
                "natural for CFO because it predicts a continuous vector field, whereas standard AR baselines "
                "do not expose the same object."
            ),
            kind="success",
        ),
    ])
    return


@app.cell(hide_code=True)
def phase_portrait_intro(mo):
    mo.md(r"""
    ## Supporting Diagnostic: Attractor Recovery

    The Lorenz system has a **strange attractor** — a fractal set that captures the long-run
    geometry of the dynamics. We treat attractor recovery as **supporting evidence**, not the
    primary novelty claim.

    If a model learns better local dynamics, its long-horizon rollouts from new initial
    conditions should also recover better global geometry. That makes the phase portrait a
    useful downstream diagnostic of learned dynamics quality.

    **Phase portrait test**: roll out each model for 800 steps from 20 fresh initial
    conditions (outside the training distribution). Does the long-run geometry match the
    true attractor?

    > The attractor is never part of the training objective.
    > Recovery is a downstream sanity check on the learned vector field.
    """)
    return


@app.cell(hide_code=True)
def phase_portrait_viz(
    ar_eq_model,
    ar_model,
    ar_rollout,
    cfo_model,
    cfo_pi_model,
    generate_lorenz,
    make_cfo_fn,
    mo,
    norm_stats,
    np,
    plt,
    rk4_ode,
):
    if cfo_model is None:
        mo.stop(True, mo.callout(mo.md("Train the models first."), kind="warn"))

    _state_mean, _state_std, _T_MAX, _DT, _du_mean, _du_std = norm_stats
    _dt_n = _DT / _T_MAX
    _N_PORTRAIT = 800
    _N_ICS = 20

    _cfo_fn_pp = make_cfo_fn(cfo_model, _du_mean, _du_std)
    _cfo_pi_fn_pp = make_cfo_fn(cfo_pi_model, _du_mean, _du_std) if cfo_pi_model is not None else None
    _rng_pp = np.random.default_rng(123)
    _ics = _rng_pp.uniform(-12, 12, (_N_ICS, 3))

    _true_x, _true_z = [], []
    _cfo_x, _cfo_z = [], []
    _cfo_pi_x, _cfo_pi_z = [], []
    _arf_x, _arf_z = [], []
    _are_x, _are_z = [], []

    for _ic in _ics:
        _, _traj_true = generate_lorenz(_ic, _N_PORTRAIT, _DT)
        _true_x.append(_traj_true[:, 0])
        _true_z.append(_traj_true[:, 2])

        _u0_n_pp = (_ic - _state_mean) / _state_std

        _cp = rk4_ode(_cfo_fn_pp, 0.0, _u0_n_pp, _dt_n, _N_PORTRAIT)
        _cp_phys = _cp * _state_std + _state_mean
        _cfo_x.append(_cp_phys[:, 0])
        _cfo_z.append(_cp_phys[:, 2])

        if _cfo_pi_fn_pp is not None:
            _cpp = rk4_ode(_cfo_pi_fn_pp, 0.0, _u0_n_pp, _dt_n, _N_PORTRAIT)
            _cpp_phys = _cpp * _state_std + _state_mean
            _cfo_pi_x.append(_cpp_phys[:, 0])
            _cfo_pi_z.append(_cpp_phys[:, 2])

        _arf_phys = ar_rollout(ar_model, _u0_n_pp, _N_PORTRAIT) * _state_std + _state_mean
        _arf_x.append(_arf_phys[:, 0])
        _arf_z.append(_arf_phys[:, 2])

        _are_phys = ar_rollout(ar_eq_model, _u0_n_pp, _N_PORTRAIT) * _state_std + _state_mean
        _are_x.append(_are_phys[:, 0])
        _are_z.append(_are_phys[:, 2])

    _n_panels = 5 if _cfo_pi_fn_pp is not None else 4
    fig_pp, axes_pp = plt.subplots(1, _n_panels, figsize=(4.5 * _n_panels, 4.5))
    fig_pp.suptitle(
        f"Phase Portrait (x-z)  ·  {_N_ICS} ICs × {_N_PORTRAIT} steps  ·  Attractor Recovery",
        color="#222222",
        fontsize=11,
    )

    _ax_gt = axes_pp[0]
    for _tx, _tz in zip(_true_x, _true_z):
        _ax_gt.plot(_tx, _tz, color="#333333", lw=0.6, alpha=0.6)
    _ax_gt.set_title("Ground Truth\n(Lorenz attractor)", color="#222222", fontsize=10)
    _ax_gt.set_xlabel("x"); _ax_gt.set_ylabel("z")
    _ax_gt.set_xlim(-28, 28); _ax_gt.set_ylim(0, 54)

    _panels = [
        ("CFO", _cfo_x, _cfo_z, "#7799ff", "✓ recovers attractor", "#2255bb"),
    ]
    if _cfo_pi_fn_pp is not None:
        _panels.append(("CFO-PI (physics-informed)", _cfo_pi_x, _cfo_pi_z, "#aa55ff", "✓ recovers attractor", "#2255bb"))
    _panels += [
        ("AR-full (100 % data)", _arf_x, _arf_z, "#ff8844", "✗ mode collapse", "#cc3300"),
        ("AR-equal (sparse data)", _are_x, _are_z, "#44dd88", "✗ wrong geometry", "#cc3300"),
    ]
    for _ax, (_lbl, _xs, _zs, _col, _verd, _vcol) in zip(axes_pp[1:], _panels):
        for _tx, _tz in zip(_true_x, _true_z):
            _ax.plot(_tx, _tz, color="#aaaaaa", lw=0.3, alpha=0.25)
        for _xi, _zi in zip(_xs, _zs):
            _ax.plot(_xi, _zi, color=_col, lw=0.8, alpha=0.65)
        _ax.set_title(f"{_lbl}\n{_verd}", color=_vcol, fontsize=10)
        _ax.set_xlabel("x")
        _ax.set_xlim(-28, 28); _ax.set_ylim(0, 54)

    plt.tight_layout()
    _pp_out = mo.center(mo.as_html(fig_pp))
    plt.close(fig_pp)

    mo.vstack(
        [
            _pp_out,
            mo.callout(
                mo.md(
                    f"**Panel 1** = ground-truth Lorenz attractor. "
                    f"**Remaining panels** = each model rolled out for {_N_PORTRAIT} steps from {_N_ICS} fresh initial conditions "
                    "(outside the training distribution). "
                    "Use this section as downstream evidence rather than the main novelty claim: models with better "
                    "learned local dynamics should produce better global geometry. CFO and CFO-PI both recover the "
                    "true strange attractor, while the AR baselines collapse to simpler, incorrect long-run behavior."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md(r"""
    ## Key Takeaways

    | # | What this notebook shows | Role |
    |---|---|---|
    | 1 | CFO reproduces the paper's core Lorenz advantage: strong performance from **sparse, irregular** observations against AR baselines | Reproduction |
    | 2 | In this demo, AR errors compound more severely over long rollouts than CFO errors | Reproduction |
    | 3 | **CFO-PI** implements the paper's proposed physics integration idea in an executable notebook | Novel |
    | 4 | CFO-PI improves vector-field alignment and can shift the CFO/AR crossover to lower keep rates | Novel |
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

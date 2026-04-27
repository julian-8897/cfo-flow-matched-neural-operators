#!/usr/bin/env python3
"""Apply notebook recovery changes: steps 6-9."""

content = open("notebook.py").read()

# ── Step 6: Replace sweep section (sweep_intro + sweep_controls + keep_rate_sweep)
# with resolution demo cells ─────────────────────────────────────────────────────

sweep_start = content.find(
    "\n\n@app.cell(hide_code=True)\ndef sweep_intro(mo):\n    mo.md(r\"\"\"\n    ## Reproduction Result: The Data-Efficiency Frontier"
)
physics_start = content.find("\n\n@app.cell(hide_code=True)\ndef physics_interp_intro(mo):")

old_sweep_section = content[sweep_start:physics_start]

new_resolution_cells = '''


@app.cell(hide_code=True)
def sweep_intro(mo):
    mo.md(r"""
    ## Temporal Generalisation: One Model, Any Resolution

    CFO learns a continuous vector field that can be queried at **any step size** without
    retraining. AR is locked to its training $\\Delta t$: a different resolution requires
    a new model.

    **Drag the slider** to vary the evaluation resolution using the single CFO model
    trained above.
    """)
    return


@app.cell(hide_code=True)
def resolution_controls(cfo_model, mo, norm_stats):
    mo.stop(
        cfo_model is None,
        mo.callout(mo.md("Train models first. Click **▶ Train Three Models** above."), kind="neutral"),
    )
    _state_mean, _state_std, _T_MAX, _DT_TRAIN, _du_mean, _du_std = norm_stats
    dt_slider = mo.ui.slider(
        start=0.005, stop=0.1, step=0.005,
        value=_DT_TRAIN, label="Evaluation Δt (s)", show_value=True,
    )
    dt_slider
    return (dt_slider,)


@app.cell(hide_code=True)
def continuous_resolution_demo(
    ar_model, ar_rollout, cfo_model, dt_slider,
    generate_lorenz, make_cfo_fn, mo, norm_stats, np, plt, rk4_ode,
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
        color="#222222", fontsize=11,
    )

    # Panel 1: x-z phase portrait
    axes_res[0].plot(_gt_fine[:, 0], _gt_fine[:, 2], color="#aaaaaa", lw=0.6, alpha=0.5, label="ground truth")
    axes_res[0].plot(_traj[:, 0], _traj[:, 2], color="#7799ff", lw=1.8, label=f"CFO (Δt={_dt_eval:.3f}s)")
    axes_res[0].plot(_ar_traj[:, 0], _ar_traj[:, 2], color="#ff8844", lw=1.5, linestyle="--", label=f"AR (Δt={_DT_TRAIN:.3f}s, fixed)")
    axes_res[0].set_xlabel("x"); axes_res[0].set_ylabel("z")
    axes_res[0].set_title("Phase portrait (x–z)", color="#333333")
    axes_res[0].legend(fontsize=8)

    # Panel 2: x over steps
    axes_res[1].plot(_steps_gt, _gt_fine[:, 0], color="#aaaaaa", lw=0.6, alpha=0.5, label="ground truth")
    axes_res[1].plot(_steps_cfo, _traj[:, 0], color="#7799ff", lw=1.8, label=f"CFO (Δt={_dt_eval:.3f}s)")
    axes_res[1].plot(_steps_ar, _ar_traj[:, 0], color="#ff8844", lw=1.5, linestyle="--", label=f"AR (Δt={_DT_TRAIN:.3f}s, fixed)")
    axes_res[1].set_xlabel("step (CFO units)"); axes_res[1].set_ylabel("x")
    axes_res[1].set_title("x-component over time", color="#333333")
    axes_res[1].legend(fontsize=8)

    plt.tight_layout()
    _res_out = mo.center(mo.as_html(fig_res))
    plt.close(fig_res)

    mo.vstack([
        _res_out,
        mo.callout(
            mo.md(
                f"CFO uses **{_n_steps} steps** at the chosen resolution. "
                f"AR is always fixed to {_ar_steps} steps at its training Δt. "
                "Change the slider: the CFO trajectory updates instantly — no retraining."
            ),
            kind="info",
        ),
    ])
    return'''

content = content[:sweep_start] + new_resolution_cells + content[physics_start:]

# ── Step 7: Delete physics_interp_intro + physics_interp_viz ─────────────────

physics_start2 = content.find("\n\n@app.cell(hide_code=True)\ndef physics_interp_intro(mo):")
phase_portrait_start2 = content.find("\n\n@app.cell(hide_code=True)\ndef phase_portrait_intro(mo):")

if physics_start2 != -1 and phase_portrait_start2 != -1:
    content = content[:physics_start2] + content[phase_portrait_start2:]
    print("Deleted physics_interp section OK")
else:
    print(f"WARNING: physics_start2={physics_start2}, phase_portrait_start2={phase_portrait_start2}")

# ── Step 8: Insert parametric CFO section before takeaways ───────────────────

takeaways_start = content.find("\n\n@app.cell(hide_code=True)\ndef takeaways(mo):")

parametric_cells = '''


@app.cell(hide_code=True)
def parametric_intro(mo):
    mo.md(r"""
    ## Novel Contribution: Parametric CFO

    One CFO model, conditioned on the Lorenz parameter $\\rho$, learns the **entire family**
    of attractors at once. At inference, setting $\\rho$ yields a continuous vector field for
    that attractor without retraining.

    This is not in the original paper — it is the notebook's primary novel contribution.
    The model conditions on normalised $\\rho$ as an extra scalar input to the network, trained
    jointly on trajectories sampled from $\\rho \\in \\{25, 28, 32, 35, 38\\}$.
    """)
    return


@app.cell(hide_code=True)
def parametric_train_controls(mo):
    _param_epochs_slider = mo.ui.slider(start=100, stop=500, step=50, value=200, label="Epochs", show_value=True)
    _param_hidden_slider = mo.ui.slider(start=64, stop=256, step=32, value=128, label="Hidden size", show_value=True)
    _param_traj_slider = mo.ui.slider(start=5, stop=20, step=5, value=10, label="Trajectories per ρ", show_value=True)
    param_train_btn = mo.ui.run_button(label="▶ Train Parametric CFO")
    mo.vstack([
        mo.hstack([_param_epochs_slider, _param_hidden_slider, _param_traj_slider]),
        param_train_btn,
    ])
    return param_train_btn, _param_epochs_slider, _param_hidden_slider, _param_traj_slider


@app.cell(hide_code=True)
def parametric_training(
    TinyODENetParam, QuinticHermiteSpline, compute_normalization,
    generate_lorenz_param, mo, nn, np, param_train_btn,
    _param_epochs_slider, _param_hidden_slider, _param_traj_slider, torch,
):
    mo.stop(
        not param_train_btn.value,
        mo.callout(mo.md("Click **▶ Train Parametric CFO** to train."), kind="neutral"),
    )
    _RHOS = [25.0, 28.0, 32.0, 35.0, 38.0]
    _DT_P = 0.025
    _N_STEPS_P = 200
    _N_SMP_P = 60
    _EPOCHS_P = _param_epochs_slider.value
    _HIDDEN_P = _param_hidden_slider.value
    _N_TRAJ_P = _param_traj_slider.value
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
        _RHO_p.append(np.full(_N_SMP_P, (_rho_v - _RHO_MEAN) / _RHO_STD, dtype=np.float32))
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
            _ib = _perm[_i:_i + _BATCH_P]
            _opt_p.zero_grad()
            _l = torch.mean((param_cfo_model(_T_t[_ib], _U_t[_ib], _RHO_t[_ib]) - _DU_t[_ib]) ** 2)
            _l.backward()
            _opt_p.step()
            _ep_loss += float(_l) * len(_ib)
        _losses_p.append(_ep_loss / _N_P)
    param_norm_stats = (_p_state_mean, _p_state_std, _T_MAX_P, _DT_P, _p_du_mean, _p_du_std, _RHO_MEAN, _RHO_STD)
    mo.callout(mo.md(f"Parametric CFO trained. Final loss: {_losses_p[-1]:.4f}"), kind="success")
    return param_cfo_model, param_norm_stats


@app.cell(hide_code=True)
def parametric_explore_controls(mo, param_cfo_model):
    mo.stop(
        param_cfo_model is None,
        mo.callout(mo.md("Train Parametric CFO first."), kind="neutral"),
    )
    rho_slider = mo.ui.slider(start=15.0, stop=50.0, step=0.5, value=28.0, label="ρ (Lorenz parameter)", show_value=True)
    rho_slider
    return (rho_slider,)


@app.cell(hide_code=True)
def vector_field_viz(
    generate_lorenz_param, make_cfo_param_fn, mo, np, param_cfo_model,
    param_norm_stats, plt, rho_slider, rk4_ode,
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

    # Grid for streamlines (x-z slice at y=0)
    _nx, _nz = 20, 20
    _xs = np.linspace(-20, 20, _nx)
    _zs = np.linspace(0, 50, _nz)
    _XX, _ZZ = np.meshgrid(_xs, _zs)
    _YY = np.zeros_like(_XX)
    _UU = np.zeros_like(_XX); _WW = np.zeros_like(_XX)
    _UU_c = np.zeros_like(_XX); _WW_c = np.zeros_like(_XX)

    for _i_g in range(_nz):
        for _j_g in range(_nx):
            _u_p = np.array([_XX[_i_g, _j_g], _YY[_i_g, _j_g], _ZZ[_i_g, _j_g]])
            _sigma, _beta = 10.0, 8.0 / 3.0
            _x, _y, _z = _u_p
            _dxyz = np.array([_sigma*(_y-_x), _x*(_rho-_z)-_y, _x*_y-_beta*_z])
            _UU[_i_g, _j_g] = _dxyz[0]; _WW[_i_g, _j_g] = _dxyz[2]
            _u_n_g = ((_u_p - _sm) / _ss).astype(np.float32)
            _v_n_g = _cfo_p_fn(0.5, _u_n_g)
            _v_p_g = _v_n_g * _ss / _T_MAX_p
            _UU_c[_i_g, _j_g] = _v_p_g[0]; _WW_c[_i_g, _j_g] = _v_p_g[2]

    fig_vf2, axes_vf2 = plt.subplots(1, 2, figsize=(12, 5))
    fig_vf2.suptitle(f"Parametric CFO: ρ = {_rho:.1f}  (x–z slice at y = 0)", color="#222222", fontsize=11)

    axes_vf2[0].streamplot(_xs, _zs, _UU, _WW, color=np.sqrt(_UU**2+_WW**2), cmap="Blues", linewidth=0.8, density=1.2)
    axes_vf2[0].plot(_gt_vf[:, 0], _gt_vf[:, 2], color="#333333", lw=1.5, alpha=0.8, label="True traj")
    axes_vf2[0].set_xlabel("x"); axes_vf2[0].set_ylabel("z")
    axes_vf2[0].set_title("True Lorenz field", color="#333333")
    axes_vf2[0].legend(fontsize=8)

    axes_vf2[1].streamplot(_xs, _zs, _UU_c, _WW_c, color=np.sqrt(_UU_c**2+_WW_c**2), cmap="Purples", linewidth=0.8, density=1.2)
    axes_vf2[1].plot(_cfo_traj_vf[:, 0], _cfo_traj_vf[:, 2], color="#7799ff", lw=1.5, alpha=0.8, label="CFO traj")
    axes_vf2[1].plot(_gt_vf[:, 0], _gt_vf[:, 2], color="#aaaaaa", lw=0.8, alpha=0.5, label="True traj")
    axes_vf2[1].set_xlabel("x"); axes_vf2[1].set_ylabel("z")
    axes_vf2[1].set_title("Parametric CFO field", color="#333333")
    axes_vf2[1].legend(fontsize=8)

    plt.tight_layout()
    _vf2_out = mo.center(mo.as_html(fig_vf2))
    plt.close(fig_vf2)
    _vf2_out
    return


@app.cell(hide_code=True)
def field_click_panel(
    go, mo, np, param_cfo_model, param_norm_stats, rho_slider,
):
    _sm, _ss, _T_MAX_p, _DT_p, _du_m, _du_s, _RHO_M, _RHO_S = param_norm_stats
    _rho = rho_slider.value
    _rho_n = float((_rho - _RHO_M) / _RHO_S)

    _nx, _nz = 30, 30
    _xs_g = np.linspace(-22, 22, _nx)
    _zs_g = np.linspace(0, 52, _nz)
    _XX_g, _ZZ_g = np.meshgrid(_xs_g, _zs_g)
    _speed = np.zeros((_nz, _nx))

    import torch as _torch
    with _torch.no_grad():
        for _i in range(_nz):
            _u_row = np.stack([
                _XX_g[_i], np.zeros(_nx), _ZZ_g[_i]
            ], axis=1)
            _u_n_row = ((_u_row - _sm) / _ss).astype(np.float32)
            _t_row = _torch.tensor([0.5] * _nx, dtype=_torch.float32)
            _u_t_row = _torch.tensor(_u_n_row)
            _rho_t_row = _torch.tensor([[_rho_n]] * _nx, dtype=_torch.float32)
            _v_n_row = param_cfo_model(_t_row, _u_t_row, _rho_t_row).numpy()
            _v_p_row = _v_n_row * _ss / _T_MAX_p
            _speed[_i] = np.sqrt(_v_p_row[:, 0]**2 + _v_p_row[:, 2]**2)

    _fig_click = go.Figure()
    _fig_click.add_trace(go.Heatmap(
        x=_xs_g, y=_zs_g, z=_speed,
        colorscale="Viridis", showscale=True,
        colorbar=dict(title="speed"),
    ))
    _fig_click.update_layout(
        title=f"Click a point to launch a trajectory (ρ = {_rho:.1f})",
        xaxis_title="x", yaxis_title="z",
        width=600, height=450,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    field_click = mo.ui.plotly(_fig_click)
    mo.vstack([
        mo.md("**Click anywhere on the heatmap** to set an initial condition (x, z) at y = 0. The trajectory appears below."),
        field_click,
    ])
    return (field_click,)


@app.cell(hide_code=True)
def click_trajectory(
    field_click, generate_lorenz_param, make_cfo_param_fn, mo,
    np, param_cfo_model, param_norm_stats, plt, rho_slider, rk4_ode,
):
    _clicked = field_click.value
    if not _clicked:
        mo.stop(True, mo.md("Click on the heatmap above to launch a trajectory."))
    _pt = _clicked[0]
    _x0_click = float(_pt["x"])
    _z0_click = float(_pt["y"])
    _y0_click = 0.0
    _ic_click = np.array([_x0_click, _y0_click, _z0_click])

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
        f"Trajectory from IC: x={_x0_click:.1f}, z={_z0_click:.1f}, y=0  |  ρ={_rho:.1f}",
        color="#222222", fontsize=10,
    )
    axes_ct[0].plot(_gt_click[:, 0], _gt_click[:, 2], color="#333333", lw=1.5, label="True")
    axes_ct[0].plot(_cfo_click[:, 0], _cfo_click[:, 2], color="#7799ff", lw=1.5, linestyle="--", label="Param CFO")
    axes_ct[0].scatter([_x0_click], [_z0_click], color="red", zorder=5, s=60, label="IC")
    axes_ct[0].set_xlabel("x"); axes_ct[0].set_ylabel("z")
    axes_ct[0].set_title("Phase portrait (x–z)", color="#333333")
    axes_ct[0].legend(fontsize=8)
    _steps_ct = np.arange(301)
    axes_ct[1].plot(_steps_ct, _gt_click[:, 0], color="#333333", lw=1.5, label="True x")
    axes_ct[1].plot(_steps_ct, _cfo_click[:, 0], color="#7799ff", lw=1.5, linestyle="--", label="Param CFO x")
    axes_ct[1].set_xlabel("step"); axes_ct[1].set_ylabel("x")
    axes_ct[1].set_title("x-component over time", color="#333333")
    axes_ct[1].legend(fontsize=8)
    plt.tight_layout()
    _ct_out = mo.center(mo.as_html(fig_click_traj))
    plt.close(fig_click_traj)
    _ct_out
    return'''

content = content[:takeaways_start] + parametric_cells + content[takeaways_start:]

# ── Step 9: Update takeaways (remove CFO-PI from the table) ──────────────────

old_takeaways_table = '''    | # | What this notebook shows | Role |
    |---|---|---|
    | 1 | CFO reproduces the paper\'s core Lorenz advantage: strong performance from **sparse, irregular** observations against AR baselines | Reproduction |
    | 2 | In this demo, AR errors compound more severely over long rollouts than CFO errors | Reproduction |
    | 3 | **CFO-PI** implements the paper\'s proposed physics integration idea in an executable notebook | Novel |
    | 4 | CFO-PI improves vector-field alignment and can shift the CFO/AR crossover to lower keep rates | Novel |
    | 5 | Attractor recovery is supporting evidence: better local dynamics lead to better global geometry | Supporting evidence |'''

new_takeaways_table = '''    | # | What this notebook shows | Role |
    |---|---|---|
    | 1 | CFO reproduces the paper\'s core Lorenz advantage: strong performance from **sparse, irregular** observations against AR baselines | Reproduction |
    | 2 | CFO learns a continuous vector field that is **resolution-agnostic**: query at any step size without retraining | Reproduction + demo |
    | 3 | **Parametric CFO** conditions on Lorenz $\\\\rho$, learning the entire family of attractors with one model | Novel |
    | 4 | The clickable vector field lets you launch trajectories from any initial condition across the attractor family | Novel demo |
    | 5 | Attractor recovery is supporting evidence: better local dynamics lead to better global geometry | Supporting evidence |'''

if old_takeaways_table in content:
    content = content.replace(old_takeaways_table, new_takeaways_table)
    print("Updated takeaways table OK")
else:
    print("WARNING: could not find takeaways table")

# Write the result
with open("notebook.py", "w") as f:
    f.write(content)

print(f"Done. New file length: {len(content)} chars")

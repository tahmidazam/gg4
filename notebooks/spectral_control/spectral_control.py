"""Closed-loop spectral control of neural activity.

Three controllers (LQG, LQI, MPC) are run against the Brain simulator to
reproduce a prescribed frequency spectrum in the population output.  The
system matrices (A, B, C, Q, R) are loaded from a saved .npz archive; the
ESTIMATOR_PATH constant in the notebook selects which estimate to use.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import lsq_linear, minimize

sys.path.insert(0, str((Path(__file__).parent.parent / "shared").resolve()))
from pgf_utils import notebook_github_url
from system_estimate import SystemEstimate

NOTEBOOK_GITHUB_URL = notebook_github_url(__file__)


# ── System model ──────────────────────────────────────────────────────────────


@dataclass
class SystemModel:
    """LGSSM with observation mean for centring.

    Operates in centred observation space: y_c = y - y_mean.
    All latent quantities are centred; physical mean = cmean @ x + base_mean.
    """

    A: np.ndarray  # (n_x, n_x)
    B: np.ndarray  # (n_x, n_u)  — B operates on centred inputs u_c = u_phys - 0.5
    C: np.ndarray  # (n_y, n_x)
    Q: np.ndarray  # (n_x, n_x)
    R: np.ndarray  # (n_y, n_y)
    y_mean: np.ndarray  # (n_y,)

    @classmethod
    def from_estimate(cls, est: SystemEstimate, y_mean: np.ndarray) -> "SystemModel":
        return cls(A=est.A, B=est.B, C=est.C, Q=est.Q, R=est.R, y_mean=y_mean)

    @property
    def n_x(self) -> int:
        return self.A.shape[0]

    @property
    def n_u(self) -> int:
        return self.B.shape[1]

    @property
    def n_y(self) -> int:
        return self.C.shape[0]

    @property
    def cmean(self) -> np.ndarray:
        return self.C.mean(axis=0)

    @property
    def base_mean(self) -> float:
        """Physical population mean at zero latent state."""
        return float(self.y_mean.mean())

    def equilibrium(self, target_centred: float) -> tuple[np.ndarray, np.ndarray]:
        """Feasible (x_ref, u_c_ref) with cmean @ x_ref = target_centred.

        Bounds for centred input: u_c = u_phys - 0.5, so u_c ∈ [-0.5, 0.5].
        Returns centred input u_c_ref; physical = u_c_ref + 0.5.
        """
        G = self.cmean @ np.linalg.solve(np.eye(self.n_x) - self.A, self.B)
        lb = np.full(self.n_u, -0.5)
        ub = np.full(self.n_u, 0.5)
        res = lsq_linear(G.reshape(1, -1), np.array([target_centred]), bounds=(lb, ub))
        u_c_ref = res.x
        x_ref = np.linalg.solve(np.eye(self.n_x) - self.A, self.B @ u_c_ref)
        return x_ref, u_c_ref


# ── Spectral target ───────────────────────────────────────────────────────────


class SpectralTarget:
    """Sum-of-sinusoids reference in physical population-mean space.

    target(t) = offset + sum_k A_k * sin(2*pi*f_k*t + phi_k)

    offset is the desired physical population mean around which the signal
    oscillates.  Controllers subtract model.base_mean internally to obtain
    the centred deviation needed for the equilibrium and LQR computations.
    """

    def __init__(self, components: list[tuple[float, float, float]], offset: float = 0.0):
        comp = np.asarray(components, dtype=float).reshape(-1, 3)
        self._freqs = comp[:, 0]
        self._amps = comp[:, 1]
        self._phases = comp[:, 2]
        self.offset = offset
        self.components = components

    def __call__(self, t: int) -> float:
        return float(self.offset + self._amps @ np.sin(2 * np.pi * self._freqs * t + self._phases))


# ── Kalman filter ─────────────────────────────────────────────────────────────


class KalmanFilter:
    """Causal Kalman filter in centred observation space."""

    def __init__(self, model: SystemModel):
        self._m = model
        self._x = np.zeros(model.n_x)
        self._P = np.eye(model.n_x)

    def reset(self) -> None:
        self._x = np.zeros(self._m.n_x)
        self._P = np.eye(self._m.n_x)

    def step(self, y: np.ndarray, u_phys_prev: np.ndarray) -> np.ndarray:
        """Predict with u_phys_prev (centred internally); update with y. Returns x_hat."""
        A, B, C, Q, R = self._m.A, self._m.B, self._m.C, self._m.Q, self._m.R
        # Centre the input: ERA+EM/CVA+EM B operates on centred inputs
        u_c = u_phys_prev - 0.5
        y_c = np.asarray(y) - self._m.y_mean
        x_p = A @ self._x + B @ u_c
        P_p = A @ self._P @ A.T + Q
        innov = y_c - C @ x_p
        S = C @ P_p @ C.T + R
        Kf = np.linalg.solve(S, C @ P_p).T
        self._x = x_p + Kf @ innov
        self._P = (np.eye(self._m.n_x) - Kf @ C) @ P_p
        return self._x.copy()

    def pop_mean(self, x_hat: np.ndarray) -> float:
        """Physical population mean from latent estimate."""
        return float(self._m.cmean @ x_hat + self._m.base_mean)


# ── LQG controller ────────────────────────────────────────────────────────────


class LQGController:
    """LQR on Kalman state with instantaneous feedforward.

    u_t = clip(u_c_ref(t) + 0.5 - K (x_hat - x_ref(t)))

    lambda_var blends the tracking cost between population-mean only
    (lambda_var=0) and full per-neuron tracking (lambda_var=1).
    """

    def __init__(
        self,
        model: SystemModel,
        q_track: float = 20.0,
        r_effort: float = 80.0,
        lambda_var: float = 0.0,
    ):
        self._m = model
        self._kf = KalmanFilter(model)
        self._K = self._solve_lqr(q_track, r_effort, lambda_var)
        self._u_prev = np.zeros(model.n_u)

    def _solve_lqr(self, q_track: float, r_effort: float, lambda_var: float) -> np.ndarray:
        cmean, C = self._m.cmean, self._m.C
        Q_mean = np.outer(cmean, cmean)
        Q_var = C.T @ C
        Q_lqr = q_track * ((1.0 - lambda_var) * Q_mean + lambda_var * Q_var) + 1e-6 * np.eye(self._m.n_x)
        R_lqr = r_effort * np.eye(self._m.n_u)
        P = solve_discrete_are(self._m.A, self._m.B, Q_lqr, R_lqr)
        return np.linalg.solve(R_lqr + self._m.B.T @ P @ self._m.B, self._m.B.T @ P @ self._m.A)

    def reset(self) -> None:
        self._kf.reset()
        self._u_prev = np.zeros(self._m.n_u)

    def __call__(self, y: np.ndarray, t: int, target: SpectralTarget) -> np.ndarray:
        x_hat = self._kf.step(y, self._u_prev)
        tgt_c = target(t) - self._m.base_mean
        x_ref, u_c_ref = self._m.equilibrium(tgt_c)
        u_c = u_c_ref - self._K @ (x_hat - x_ref)
        self._u_prev = np.clip(u_c + 0.5, 0.0, 1.0)
        return self._u_prev

    @property
    def last_kf_mean(self) -> float:
        return self._kf.pop_mean(self._kf._x)


# ── LQI controller ────────────────────────────────────────────────────────────


class LQIController:
    """LQG augmented with integral action on the centred output error.

    q_int must be small for sinusoidal references: a large integrator gain
    accumulates sinusoidal tracking error and suppresses the oscillatory
    feedforward.  q_int ≈ 0.002 provides slow DC-bias correction without
    disturbing spectral tracking.
    """

    def __init__(
        self,
        model: SystemModel,
        q_track: float = 20.0,
        r_effort: float = 200.0,
        q_int: float = 0.002,
        lambda_var: float = 0.0,
    ):
        self._m = model
        self._kf = KalmanFilter(model)
        self._Kx, self._Ki = self._solve_lqi(q_track, r_effort, q_int, lambda_var)
        self._sigma = 0.0
        self._u_prev = np.zeros(model.n_u)

    def _solve_lqi(
        self, q_track: float, r_effort: float, q_int: float, lambda_var: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n, m = self._m.n_x, self._m.n_u
        c, C = self._m.cmean, self._m.C
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = self._m.A
        A_aug[n, :n] = c
        A_aug[n, n] = 1.0
        B_aug = np.zeros((n + 1, m))
        B_aug[:n] = self._m.B
        Q_mean = np.outer(c, c)
        Q_var = C.T @ C
        Q_aug = np.zeros((n + 1, n + 1))
        Q_aug[:n, :n] = q_track * ((1.0 - lambda_var) * Q_mean + lambda_var * Q_var) + 1e-6 * np.eye(n)
        Q_aug[n, n] = q_int
        R_lqr = r_effort * np.eye(m)
        P = solve_discrete_are(A_aug, B_aug, Q_aug, R_lqr)
        K = np.linalg.solve(R_lqr + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_aug)
        return K[:, :n], K[:, n]

    def reset(self) -> None:
        self._kf.reset()
        self._sigma = 0.0
        self._u_prev = np.zeros(self._m.n_u)

    def __call__(self, y: np.ndarray, t: int, target: SpectralTarget) -> np.ndarray:
        tgt_c = target(t) - self._m.base_mean
        x_hat = self._kf.step(y, self._u_prev)
        x_ref, u_c_ref = self._m.equilibrium(tgt_c)
        self._sigma += float(self._m.cmean @ x_hat - tgt_c)
        u_c = u_c_ref - self._Kx @ (x_hat - x_ref) - self._Ki * self._sigma
        self._u_prev = np.clip(u_c + 0.5, 0.0, 1.0)
        return self._u_prev

    @property
    def last_kf_mean(self) -> float:
        return self._kf.pop_mean(self._kf._x)


# ── MPC controller ────────────────────────────────────────────────────────────


class MPCController:
    """Receding-horizon QP with box constraints.

    Optimises over physical inputs u_phys ∈ [0, 1] but maps through centred
    inputs u_c = u_phys - 0.5 for the state-space predictions (matching the
    ERA+EM/CVA+EM B matrix convention).
    """

    def __init__(
        self,
        model: SystemModel,
        horizon: int = 30,
        q_track: float = 20.0,
        r_u: float = 10.0,
        r_du: float = 10.0,
        lambda_var: float = 0.0,
    ):
        self._m = model
        self._H = horizon
        self._lambda_var = lambda_var
        self._kf = KalmanFilter(model)
        self._u_prev = np.zeros(model.n_u)
        self._u_seq = np.full(horizon * model.n_u, 0.5)  # warm start at midpoint
        self._solver = self._build_solver(horizon, q_track, r_u, r_du, lambda_var)

    def _build_solver(self, H: int, q_track: float, r_u: float, r_du: float, lambda_var: float):
        n, m = self._m.n_x, self._m.n_u
        N = self._m.n_y
        A, B, C, cmean = self._m.A, self._m.B, self._m.C, self._m.cmean

        # State prediction matrices (condensed form)
        Phi = np.zeros((H * n, n))
        Ak = np.eye(n)
        for k in range(H):
            Ak = A @ Ak
            Phi[k * n : (k + 1) * n] = Ak

        Gamma = np.zeros((H * n, H * m))
        for k in range(H):
            for j in range(k + 1):
                Gamma[k * n : (k + 1) * n, j * m : (j + 1) * m] = (
                    np.linalg.matrix_power(A, k - j) @ B
                )

        # Output projection operators
        Cbar_mean = np.zeros((H, H * n))  # population mean
        Cbar_full = np.zeros((H * N, H * n))  # all neurons
        for k in range(H):
            Cbar_mean[k, k * n : (k + 1) * n] = cmean
            Cbar_full[k * N : (k + 1) * N, k * n : (k + 1) * n] = C

        Ksys_mean = Cbar_mean @ Gamma  # (H, H*m)
        Ksys_full = Cbar_full @ Gamma  # (H*N, H*m)

        # Input-rate difference operator
        D = np.zeros((H * m, H * m))
        for k in range(H):
            D[k * m : (k + 1) * m, k * m : (k + 1) * m] = np.eye(m)
            if k > 0:
                D[k * m : (k + 1) * m, (k - 1) * m : k * m] = -np.eye(m)

        # Constant Hessian (target-independent)
        K_track = (1.0 - lambda_var) * Ksys_mean.T @ Ksys_mean + lambda_var * Ksys_full.T @ Ksys_full
        Hess = 2.0 * (q_track * K_track + r_du * D.T @ D + r_u * np.eye(H * m))

        # u_c = u_phys - 0.5: convert free-run prediction to u_phys optimisation
        u_mean_tile = np.tile(np.full(m, 0.5), H)  # (H*m,)
        # Offset from mean u: Cbar_mean @ Gamma @ u_mean_tile
        Ksys_mean_offset = Ksys_mean @ u_mean_tile   # (H,)  -- constant
        Ksys_full_offset = Ksys_full @ u_mean_tile   # (H*N,)

        bounds_phys = [(0.0, 1.0)] * (H * m)

        def solve(
            x0: np.ndarray,
            u_phys_prev: np.ndarray,
            tgt_c_seq: np.ndarray,
            u_seq_prev: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            # Free-run (zero centred input = u_phys=0.5) prediction from x0
            free_mean = Cbar_mean @ (Phi @ x0)  # (H,)
            free_full = Cbar_full @ (Phi @ x0)  # (H*N,)

            # Adjusted error: optimising over u_phys, with Gamma @ u_c = Gamma @ (u_phys - 0.5)
            e0_mean = free_mean - Ksys_mean_offset - tgt_c_seq        # (H,)
            e0_full = free_full - Ksys_full_offset - np.tile(tgt_c_seq, N)  # (H*N,)  approx

            s0 = np.zeros(H * m)
            s0[:m] = u_phys_prev

            # u_ref: physical equilibrium for mean of horizon target
            _, u_c_ref = self._m.equilibrium(float(tgt_c_seq.mean()))
            U_ref = np.tile(u_c_ref + 0.5, H)  # physical equilibrium

            g = 2.0 * (
                q_track * ((1.0 - lambda_var) * Ksys_mean.T @ e0_mean
                           + lambda_var       * Ksys_full.T @ e0_full)
                - r_du * D.T @ s0
                - r_u  * U_ref
            )

            def cost(z: np.ndarray) -> float:
                return float(0.5 * z @ Hess @ z + g @ z)

            def jac(z: np.ndarray) -> np.ndarray:
                return Hess @ z + g

            u0 = np.roll(u_seq_prev, -m)
            u0[-m:] = u_seq_prev[-m:]
            res = minimize(cost, u0, jac=jac, method="L-BFGS-B", bounds=bounds_phys)
            return res.x[:m], res.x

        return solve

    def reset(self) -> None:
        self._kf.reset()
        self._u_prev = np.zeros(self._m.n_u)
        self._u_seq = np.full(self._H * self._m.n_u, 0.5)

    def __call__(self, y: np.ndarray, t: int, target: SpectralTarget) -> np.ndarray:
        x_hat = self._kf.step(y, self._u_prev)
        tgt_c_seq = np.array([target(t + k) - self._m.base_mean for k in range(self._H)])
        u, self._u_seq = self._solver(x_hat, self._u_prev, tgt_c_seq, self._u_seq)
        self._u_prev = np.clip(u, 0.0, 1.0)
        return self._u_prev

    @property
    def last_kf_mean(self) -> float:
        return self._kf.pop_mean(self._kf._x)


# ── No-input baseline ─────────────────────────────────────────────────────────


class NoInputController:
    """Zero input at every step: uncontrolled baseline."""

    def __init__(self, n_u: int):
        self._n_u = n_u

    def reset(self) -> None:
        pass

    def __call__(self, y: np.ndarray, t: int, target: SpectralTarget) -> np.ndarray:
        return np.zeros(self._n_u)

    @property
    def last_kf_mean(self) -> float:
        return float("nan")


# ── Closed-loop runner ────────────────────────────────────────────────────────


def run_closed_loop(
    brain_factory,
    model: SystemModel,
    controller,
    target: SpectralTarget,
    T: int,
    n_burnin: int = 200,
) -> dict:
    """Run one closed-loop trial. Returns Y (T,n_y), U (T,n_u), y_bar (T,), y_kf (T,)."""
    controller.reset()
    brain = brain_factory()

    burn_Y = []
    for _ in range(n_burnin):
        burn_Y.append(np.array(brain.measure()))
        brain.next_state()
    model.y_mean = np.mean(burn_Y, axis=0)
    controller.reset()

    Y = np.empty((T, model.n_y))
    U = np.empty((T, model.n_u))
    y_kf = np.empty(T)
    for t in range(T):
        y = np.array(brain.measure())
        u = controller(y, t, target)
        Y[t] = y
        U[t] = u
        kf_mean = controller.last_kf_mean
        y_kf[t] = kf_mean if not np.isnan(kf_mean) else float(y.mean())
        brain.next_state(u.tolist())

    return {"Y": Y, "U": U, "y_bar": Y.mean(axis=1), "y_kf": y_kf}


# ── Hyperparameter search ─────────────────────────────────────────────────────


def random_search(
    brain_factory,
    model: SystemModel,
    target: SpectralTarget,
    make_ctrl,
    param_space: dict,
    n_trials: int = 8,
    T_opt: int = 200,
    warmup: int = 50,
    seed: int = 1,
    n_burnin: int = 200,
) -> tuple[dict, float]:
    """Log-uniform random search for controller hyperparameters.

    Scores each candidate by tracking RMS of y_bar against target (physical),
    skipping the initial warmup steps.

    Returns (best_params, best_rms_error).
    """
    rng = np.random.default_rng(seed)
    best_params: dict = {}
    best_err = float("inf")
    for _ in range(n_trials):
        params = {
            k: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            for k, (lo, hi) in param_space.items()
        }
        ctrl = make_ctrl(params)
        res = run_closed_loop(brain_factory, model, ctrl, target, T=T_opt, n_burnin=n_burnin)
        t_arr = np.arange(T_opt)
        ref = np.array([target(t) for t in t_arr])
        e = ref[warmup:] - res["y_bar"][warmup:]
        err = float(np.sqrt(np.mean(e**2)))
        if err < best_err:
            best_err, best_params = err, params
    return best_params, best_err


# ── Figure ────────────────────────────────────────────────────────────────────


def make_spectral_control_figure(
    results: dict[str, dict],
    result_unc: dict,
    target: SpectralTarget,
    model: SystemModel,
    controller_colours: dict[str, str] | None = None,
    figsize: tuple[float, float] = (12.0, 9.0),
    n_fft: int = 512,
) -> object:
    """Four-row figure for LQG / LQI / MPC spectral control.

    Row 1: physical population mean (y_bar only) vs uncontrolled baseline and
           target reference. Shared x-axis with rows 2 and 3.
    Row 2: observation heatmap (16 neurons × T time steps). Shared x-axis.
    Row 3: input heatmap (n_u channels × T time steps). Shared x-axis.
    Row 4: per-neuron amplitude spectral density heatmap (neuron × frequency),
           log x-axis in rad/sample with π-fraction ticks matching freq_map.

    Two separate GridSpec objects give an explicit gap between the time-domain
    section (rows 1–3) and the frequency section (row 4), ensuring axis labels
    are never hidden.  A single shared vertical colorbar is placed on the far
    right.  The legend sits below the figure.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    names = list(results.keys())
    n_cols = len(names)
    colours = controller_colours or {n: f"C{i}" for i, n in enumerate(names)}
    target_freqs_rad = target._freqs * 2 * np.pi

    xticks_rad = [np.pi / 128, np.pi / 32, np.pi / 8, np.pi / 2, np.pi]
    xticklabels_rad = [
        r"$\frac{\pi}{128}$", r"$\frac{\pi}{32}$",
        r"$\frac{\pi}{8}$",   r"$\frac{\pi}{2}$", r"$\pi$",
    ]
    freqs_rad = np.fft.rfftfreq(n_fft) * 2 * np.pi

    fig = plt.figure(figsize=figsize)

    # Absolute figure-coordinate bounds for the two sections.
    # L/R leave room for y-axis labels (left) and the shared colorbar (right).
    L, R = 0.07, 0.91

    # Time-domain section: pop-mean + obs heatmap + inputs (rows share x-axis).
    # Pop-mean ratio raised to 3.0 for more vertical space on that row.
    # bottom=0.44 leaves a 12 % gap above gs_f for the "t (steps)" label.
    gs_t = gridspec.GridSpec(
        3, n_cols,
        left=L, right=R, top=0.93, bottom=0.44,
        height_ratios=[3.0, 2.0, 0.7],
        hspace=0.40, wspace=0.10,
    )

    # Frequency section: PSD heatmap — half the original height (0.12 vs 0.25).
    # bottom=0.20 keeps 20 % below for π-fraction tick labels, ω xlabel, legend.
    gs_f = gridspec.GridSpec(
        1, n_cols,
        left=L, right=R, top=0.32, bottom=0.20,
        wspace=0.10,
    )

    l_unc = l_ref = None
    im_last = None

    for col, name in enumerate(names):
        res = results[name]
        Y = res["Y"]
        y_bar = res["y_bar"]
        T = len(y_bar)
        t_axis = np.arange(T)
        c = colours[name]
        n_y = Y.shape[1]
        n_u = res["U"].shape[1]

        # ── Row 0: population mean ────────────────────────────────────────────
        ax_y = fig.add_subplot(gs_t[0, col])
        ref = np.array([target(t) for t in t_axis])
        l_unc_line, = ax_y.plot(
            np.arange(len(result_unc["y_bar"]))[:T],
            result_unc["y_bar"][:T],
            color="grey", lw=0.9, ls="--", alpha=0.7,
        )
        l_ctrl, = ax_y.plot(t_axis, y_bar, lw=1.3, color=c)
        l_ref_line, = ax_y.plot(t_axis, ref, color="k", ls="--", lw=0.9)
        ax_y.set_title(name)
        ax_y.tick_params(labelbottom=False)
        if col == 0:
            ax_y.set_ylabel(r"$\bar{y}(t)$")
            l_unc = l_unc_line
            l_ref = l_ref_line

        # ── Row 1: observation heatmap ────────────────────────────────────────
        ax_obs = fig.add_subplot(gs_t[1, col], sharex=ax_y)
        ax_obs.imshow(
            Y.T,
            aspect="auto", origin="lower", interpolation="nearest",
            extent=[0, T, 0.5, n_y + 0.5],
            cmap="viridis", rasterized=True,
        )
        ax_obs.set_yticks([1, 8, n_y])
        ax_obs.tick_params(labelbottom=False)
        if col == 0:
            ax_obs.set_ylabel("neuron")

        # ── Row 2: input heatmap ──────────────────────────────────────────────
        ax_u = fig.add_subplot(gs_t[2, col], sharex=ax_y)
        ax_u.imshow(
            res["U"].T,
            aspect="auto", origin="lower", interpolation="nearest",
            extent=[0, T, -0.5, n_u - 0.5],
            vmin=0, vmax=1, cmap="viridis", rasterized=True,
        )
        ax_u.set_yticks(range(n_u))
        ax_u.set_xlabel(r"$t$ (steps)")
        if col == 0:
            ax_u.set_yticklabels([rf"$u_{k}$" for k in range(n_u)])
            ax_u.set_ylabel("input")
        else:
            ax_u.set_yticklabels([])

        # ── Row 3: neuron × frequency PSD heatmap ────────────────────────────
        ax_freq = fig.add_subplot(gs_f[0, col])
        psd = np.zeros((len(freqs_rad), n_y))
        for i in range(n_y):
            s = Y[-n_fft:, i] - Y[-n_fft:, i].mean()
            psd[:, i] = (2.0 / n_fft) * np.abs(np.fft.rfft(s, n=n_fft))
        im = ax_freq.pcolormesh(
            freqs_rad[1:], np.arange(1, n_y + 1), psd[1:].T,
            cmap="hot", shading="auto", rasterized=True,
        )
        for f_rad in target_freqs_rad:
            ax_freq.axvline(f_rad, color="k", lw=0.9, ls="--")
        ax_freq.set_xscale("log")
        ax_freq.set_xlim(freqs_rad[1], np.pi)
        ax_freq.set_xticks(xticks_rad)
        ax_freq.set_xticklabels(xticklabels_rad)
        ax_freq.set_yticks([1, 8, n_y])
        ax_freq.set_xlabel(r"$\omega$ (rad\,sample$^{-1}$)")
        if col == 0:
            ax_freq.set_ylabel("neuron")
        im_last = im

    # ── Single shared vertical colorbar on the far right ─────────────────────
    # Positioned to span the frequency section height exactly.
    cax = fig.add_axes([R + 0.01, 0.20, 0.015, 0.12])
    fig.colorbar(im_last, cax=cax, label="amplitude")

    # ── Legend below the figure ───────────────────────────────────────────────
    legend_handles, legend_labels = [], []
    for name in names:
        legend_handles.append(mlines.Line2D([], [], color=colours[name], lw=1.5))
        legend_labels.append(name)
    if l_unc is not None:
        legend_handles.append(l_unc)
        legend_labels.append("uncontrolled")
    if l_ref is not None:
        legend_handles.append(l_ref)
        legend_labels.append("target")

    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )

    return fig


# ── Spectral metrics ──────────────────────────────────────────────────────────


def spectral_metrics(
    results: dict[str, dict],
    target: SpectralTarget,
    n_fft: int = 512,
) -> object:
    """RMS amplitude error and off-target power for each controller."""
    import pandas as pd

    target_freqs = target._freqs
    target_amps = target._amps
    bins = np.clip(np.round(target_freqs * n_fft).astype(int), 0, n_fft // 2)
    rows = []
    for name, res in results.items():
        y_bar = res["y_bar"]
        sig = y_bar[-n_fft:] - y_bar[-n_fft:].mean()
        spec = (2.0 / n_fft) * np.abs(np.fft.rfft(sig, n=n_fft))
        amp_err = spec[bins] - target_amps
        off_mask = np.ones(len(spec), dtype=bool)
        off_mask[bins] = False
        rows.append({
            "controller": name,
            "spectral_rms_error": round(float(np.sqrt((amp_err**2).mean())), 4),
            "spectral_max_error": round(float(np.max(np.abs(amp_err))), 4),
            "off_target_power": round(
                float(spec[off_mask].mean() / (target_amps.mean() + 1e-12)), 4
            ),
        })
    return pd.DataFrame(rows).set_index("controller")

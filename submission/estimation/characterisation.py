"""Characterisation of identified models and the brain's neural response.

Groups three offline analyses that validate or probe the estimated system:

* frequency response — analytic ``C(zI−A)^{-1}B`` vs. the DFT of empirical
  Markov parameters (the Bode / frequency-map comparison);
* one-step-ahead Kalman prediction and its R² (observation reconstruction);
* per-neuron amplitude spectra under sustained sinusoidal drive (neural
  correlates of muscle selection).
"""

from __future__ import annotations

import numpy as np
from GG4 import Brain

from .system_estimate import SystemEstimate


# ── Frequency response ──────────────────────────────────────────────────────


def analytic_transfer_function(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """Evaluate H(e^{jω}) = z·C(zI − A)^{−1}B at each angular frequency.

    The leading z aligns with the empirical DFT convention where the lag-0
    index holds h(1) = CB, so the DFT gives z·H(z) rather than H(z).

    Returns complex array of shape (len(freqs), q, p).
    """
    n = A.shape[0]
    eye = np.eye(n)
    H = np.empty((len(freqs), C.shape[0], B.shape[1]), dtype=complex)
    for k, omega in enumerate(freqs):
        z = np.exp(1j * omega)
        H[k] = z * (C @ np.linalg.solve(z * eye - A, B))
    return H


def empirical_transfer_function(
    markov: np.ndarray,
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical H(e^{jω}) via DFT of Markov parameters.

    Parameters
    ----------
    markov : (N, q, p) Markov parameter tensor
    n_fft  : FFT length (zero-pads if larger than N)

    Returns
    -------
    freqs : (n_fft // 2 + 1,) angular frequency grid in [0, π] rad/sample
    H_emp : (n_fft // 2 + 1, q, p) complex empirical response
    """
    if n_fft is None:
        n_fft = markov.shape[0]
    H_emp = np.fft.rfft(markov, n=n_fft, axis=0)
    freqs = np.fft.rfftfreq(n_fft) * 2 * np.pi
    return freqs, H_emp


# Legacy names used by the bode and freq_map notebooks.
compute_analytic_bode = analytic_transfer_function
compute_empirical_bode = empirical_transfer_function
compute_analytic_response = analytic_transfer_function
compute_empirical_response = empirical_transfer_function


# ── One-step-ahead Kalman prediction ──────────────────────────────────────────


def one_step_ahead(
    Y_c: np.ndarray,
    U_c: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    p0_scale: float = 100.0,
) -> np.ndarray:
    """Return one-step-ahead predicted observations of shape (T, n_y).

    Y_hat[t] = C x_hat(t | t-1).  At t=0 the prediction is C @ 0 = 0 (no
    prior), so the first `t_skip` rows should be discarded before scoring.
    """
    T, n_y = Y_c.shape
    n_x = A.shape[0]
    eye = np.eye(n_x)

    x = np.zeros(n_x)
    P = p0_scale * np.eye(n_x)
    Y_hat = np.zeros((T, n_y))

    for t in range(T):
        # Predict x(t | t-1) from x(t-1 | t-1) and u(t-1)
        x_pred = (A @ x + B @ U_c[t - 1]) if t > 0 else np.zeros(n_x)
        P_pred = (A @ P @ A.T + Q) if t > 0 else P

        Y_hat[t] = C @ x_pred

        # Update with y_c(t)
        innov = Y_c[t] - C @ x_pred
        S = C @ P_pred @ C.T + R
        K = np.linalg.solve(S.T, (P_pred @ C.T).T).T
        x = x_pred + K @ innov
        P = (eye - K @ C) @ P_pred

    return Y_hat


def r2_per_channel(
    Y_c: np.ndarray,
    Y_hat: np.ndarray,
    t_skip: int,
) -> np.ndarray:
    """R² per output channel (shape: n_y) after discarding the first t_skip steps."""
    Y_eval = Y_c[t_skip:]
    H_eval = Y_hat[t_skip:]
    ss_res = np.sum((Y_eval - H_eval) ** 2, axis=0)
    ss_tot = np.sum((Y_eval - Y_eval.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def r2_frobenius(
    Y_c: np.ndarray,
    Y_hat: np.ndarray,
    t_skip: int,
) -> float:
    """R² computed from Frobenius norm, summing over both time and channels."""
    Y_eval = Y_c[t_skip:]
    H_eval = Y_hat[t_skip:]
    ss_res = float(np.sum((Y_eval - H_eval) ** 2))
    ss_tot = float(np.sum((Y_eval - Y_eval.mean(axis=0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def run_reconstruction_trial(
    seed: int,
    n_steps: int,
    n_burnin: int,
    estimates: list[SystemEstimate],
    t_skip: int,
) -> dict[str, tuple[np.ndarray, float]]:
    """Run one test trial and evaluate all estimators.

    Returns a dict keyed by ``est.name``, each value being
    ``(r2_per_channel, r2_frobenius)`` with shapes ``(n_y,)`` and ``()``.
    """
    brain = Brain(random_seed=seed)
    rng = np.random.default_rng(seed * 100 + 7)
    n_u = brain.input_dim
    n_y = len(brain.measure())

    for _ in range(n_burnin):
        brain.next_state(rng.random(n_u))

    Y = np.empty((n_steps, n_y))
    U = np.empty((n_steps, n_u))
    for t in range(n_steps):
        Y[t] = brain.measure()
        u = rng.random(n_u)
        U[t] = u
        brain.next_state(u)

    Y_c = Y - Y.mean(axis=0)
    U_c = U - U.mean(axis=0)

    results: dict[str, tuple[np.ndarray, float]] = {}
    for est in estimates:
        Y_hat = one_step_ahead(Y_c, U_c, est.A, est.B, est.C, est.Q, est.R)
        results[est.name] = (
            r2_per_channel(Y_c, Y_hat, t_skip),
            r2_frobenius(Y_c, Y_hat, t_skip),
        )
    return results


# ── Neural correlates of muscle selection ─────────────────────────────────────


def run_neural_trial(
    freq: float,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    *,
    offset: float,
    amplitude: float,
    t_trial: int,
    t_burnin: int,
) -> np.ndarray:
    """Run one open-loop sinusoidal trial; return neural observations.

    Drives the brain with:
        u0 = offset + amplitude * u0_scale * sin(2π freq t)
        u1 = offset + amplitude * u1_scale * sin(2π freq t)

    Returns Y of shape (t_trial, 16).
    """
    brain = Brain(random_seed=seed)
    for _ in range(t_burnin):
        brain.next_state()

    Y = np.empty((t_trial, 16))
    for t in range(t_trial):
        s = amplitude * np.sin(2 * np.pi * freq * t)
        brain.next_state([offset + s * u0_scale, offset + s * u1_scale])
        Y[t] = np.array(brain.measure())
    return Y


def _neural_trial_job(
    bi: int,
    si: int,
    freq: float,
    u0_scale: float,
    u1_scale: float,
    seed: int,
    **kw,
) -> tuple[int, int, np.ndarray]:
    """Joblib wrapper — returns (bi, si, Y) for in-order accumulation."""
    Y = run_neural_trial(freq, u0_scale, u1_scale, seed, **kw)
    return bi, si, Y


def compute_psd_matrix(Y: np.ndarray, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum for each neuron.

    Uses the last n_fft samples of Y to reduce transient contamination.

    Returns
    -------
    freqs : (n_fft // 2 + 1,) in cyc/step
    psd   : (n_fft // 2 + 1, n_y) amplitude, normalised as (2/n_fft)|FFT|
    """
    n_y = Y.shape[1]
    seg = Y[-n_fft:]
    freqs = np.fft.rfftfreq(n_fft)
    psd = np.zeros((len(freqs), n_y))
    for i in range(n_y):
        s = seg[:, i] - seg[:, i].mean()
        psd[:, i] = (2.0 / n_fft) * np.abs(np.fft.rfft(s, n=n_fft))
    return freqs, psd


def amplitude_profiles(
    freqs: np.ndarray,
    psd_data: np.ndarray,
    bands: list[tuple[float, str, str]],
) -> np.ndarray:
    """Amplitude at each band's ω_sel per neuron, mean across seeds.

    Returns A of shape (n_bands, n_y).
    """
    mean_psd = psd_data.mean(axis=1)  # (n_bands, n_freq, n_y)
    n_bands = len(bands)
    n_y = psd_data.shape[-1]
    A = np.zeros((n_bands, n_y))
    for bi, (freq, _, _) in enumerate(bands):
        bin_idx = int(np.argmin(np.abs(freqs - freq)))
        A[bi] = mean_psd[bi, bin_idx, :]
    return A

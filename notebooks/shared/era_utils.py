"""Shared ERA data-collection utilities.

Both the latent-dimension selection notebook and the ERA identification notebook
depend on these functions, so they live here rather than in either notebook's
own module folder.
"""

from __future__ import annotations

import numpy as np
from GG4 import Brain
from tqdm.auto import tqdm


def collect_markov_parameters(seed: int, p: int, n_markov: int) -> np.ndarray:
    """Return Markov parameter tensor of shape (n_markov, q, p).

    Uses impulse-subtraction: two Brain instances share the same random seed
    so additive noise cancels exactly, yielding noise-free h(k).

    markov[k, :, j] = h(k+1)[:, j] = C A^k B e_j  (k is 0-indexed).

    A throw-away probe determines q so that brain_ref and every brain_j all
    start from the identical PRNG state (no prior measure() call).
    """
    q = len(Brain(random_seed=seed).measure())

    brain_ref = Brain(random_seed=seed)
    y_ref = np.empty((n_markov, q))
    for k in range(n_markov):
        brain_ref.next_state()
        y_ref[k] = brain_ref.measure()

    markov = np.zeros((n_markov, q, p))
    with tqdm(total=p, desc="Impulse responses") as pbar:
        for j in range(p):
            brain_j = Brain(random_seed=seed)
            u = np.zeros(p)
            u[j] = 1.0
            for k in range(n_markov):
                if k == 0:
                    brain_j.next_state(u)
                else:
                    brain_j.next_state()
                markov[k, :, j] = np.array(brain_j.measure()) - y_ref[k]
            pbar.update(1)
            pbar.set_postfix(channel=j)
    return markov


def estimate_markov_ols(
    seed: int,
    drive_seed: int,
    p: int,
    n_markov: int,
    n_samples: int,
    n_burnin: int,
) -> np.ndarray:
    """Estimate Markov parameters via OLS from a single honest time series.

    Drives the Brain with i.i.d. Uniform[0, 1] inputs for n_burnin + n_samples
    steps. Fits the FIR model

        y(t) ≈ Σ_{k=0}^{n_markov-1}  h(k) u(t − k)

    by ordinary least squares, where h(k) ≈ C A^k B.  No noise cancellation
    is performed: the noise appears as regression residuals.

    Parameters
    ----------
    seed       : Brain random seed.
    drive_seed : RNG seed for the i.i.d. input sequence.
    p          : Number of input channels (n_u).
    n_markov   : Number of Markov parameters to estimate.
    n_samples  : Recording length in steps (after burn-in).
    n_burnin   : Burn-in steps discarded before recording.

    Returns
    -------
    markov : ndarray, shape (n_markov, q, p)
        markov[k] ≈ C A^k B  (0-indexed, so markov[0] ≈ CB).
    """
    rng = np.random.default_rng(drive_seed)
    brain = Brain(random_seed=seed)
    q = len(brain.measure())

    # burn-in: drive the Brain to steady state with the same RNG stream
    for _ in range(n_burnin):
        brain.next_state(rng.uniform(0.0, 1.0, size=p))

    # record n_samples steps of input/output
    U = np.empty((n_samples, p))
    Y = np.empty((n_samples, q))
    for t in range(n_samples):
        u_t = rng.uniform(0.0, 1.0, size=p)
        brain.next_state(u_t)
        U[t] = u_t
        Y[t] = brain.measure()

    # build block-Toeplitz regressor (vectorised over lags):
    #   Phi[i] = [U[t], U[t-1], ..., U[t-n_markov+1]]  where t = i + n_markov - 1
    T_valid = n_samples - n_markov + 1
    Phi = np.hstack([U[n_markov - 1 - k : n_samples - k] for k in range(n_markov)])
    # shape: (T_valid, n_markov * p)

    Y_valid = Y[n_markov - 1 :]  # shape: (T_valid, q)

    # solve OLS; Theta has shape (n_markov * p, q)
    Theta, _, _, _ = np.linalg.lstsq(Phi, Y_valid, rcond=None)

    # extract: markov[k] = h(k) has shape (q, p)
    markov = np.empty((n_markov, q, p))
    for k in range(n_markov):
        markov[k] = Theta[k * p : (k + 1) * p].T
    return markov


def build_hankel(
    markov: np.ndarray, n_rows: int, n_cols: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build block Hankel matrices H0 and H1 from Markov parameters.

    H0 block (i, j) = h(i+j+1) = markov[i+j]
    H1 block (i, j) = h(i+j+2) = markov[i+j+1]

    Requires markov.shape[0] >= n_rows + n_cols + 1.
    """
    n_markov, q, p = markov.shape
    H0 = np.zeros((n_rows * q, n_cols * p))
    H1 = np.zeros((n_rows * q, n_cols * p))
    for i in range(n_rows):
        for j in range(n_cols):
            H0[i * q : (i + 1) * q, j * p : (j + 1) * p] = markov[i + j]
            H1[i * q : (i + 1) * q, j * p : (j + 1) * p] = markov[i + j + 1]
    return H0, H1

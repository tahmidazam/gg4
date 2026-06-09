"""Shared ERA data-collection utilities.

Both the model-order selection notebook and the ERA identification notebook
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

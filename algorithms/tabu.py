import numpy as np
import time
import random

import sys, os
sys.path.append(os.path.abspath(".."))
from functions import *


def _apply_insert_inplace_np(sigma: np.ndarray, i: int, j: int) -> int:
    """
    Insert move in-place on numpy array sigma.
    Moves element at index i to index j (final position).
    Returns moved element x.
    Cost: O(|i-j|) due to shifting.
    """
    x = int(sigma[i])
    if j > i:
        # shift left the block (i+1..j) into (i..j-1)
        sigma[i:j] = sigma[i + 1 : j + 1]
        sigma[j] = x
    else:
        # shift right the block (j..i-1) into (j+1..i)
        sigma[j + 1 : i + 1] = sigma[j:i]
        sigma[j] = x
    return x


def tabu_search_insert(
    W,                    # Weight matrix of the LOP (NxN)
    sigma0,               # Initial permutation
    max_iters=5000,       # Maximum number of tabu search iterations
    max_no_improve=800,   # Stop if best solution not improved for this many iterations
    tenure=None,          # Tabu tenure length (if None, set proportional to n)
    tenure_randomize=True,# Randomize tabu tenure to avoid cycles
    window_k=None,        # Restrict insert positions to a window around i (speed-up)
    seed=0,               # Random seed for reproducibility
):
    """
    Tabu Search for LOP with INSERT neighborhood.

    Returns: best_sigma(list[int]), best_f(int), elapsed_time(float)
    """
    start = time.perf_counter()

    W = np.asarray(W)
    n = W.shape[0]
    if W.shape[1] != n:
        raise ValueError("W must be square NxN")

    rng = random.Random(seed)

    # Precompute antisymmetric matrix A[x,y] = W[x,y] - W[y,x]
    # Then:
    #   move x right across segment S: delta = - sum(A[x, S])
    #   move x left  across segment S: delta = + sum(A[x, S])
    A = W - W.T

    # Parameters
    base_tenure = max(7, n // 10) if tenure is None else max(1, int(tenure))
    if window_k is None:
        win = None if n <= 200 else max(20, n // 10)
    else:
        win = max(1, int(window_k))

    # State as numpy array for fast slicing/shifting
    sigma = np.array(list(sigma0), dtype=np.int32, copy=True)
    cur_f = objective_function(W, sigma.tolist())

    best_sigma = sigma.copy()
    best_f = cur_f

    # Tabu matrix: tabu_expire[x, j] = iteration when tabu expires
    # (x is element ID, j is position)
    tabu_expire = np.zeros((n, n), dtype=np.int32)

    no_improve = 0

    NEG_INF = -10**18  # for masking

    for it in range(1, max_iters + 1):
        # tenure (optionally variable)
        if tenure_randomize:
            lo = max(2, base_tenure // 2)
            hi = max(lo, int(base_tenure * 1.5))
            t = rng.randint(lo, hi)
        else:
            t = base_tenure

        best_delta = NEG_INF
        best_i = -1
        best_j = -1
        best_x = -1

        # Explore best-improvement (with window if provided)
        for i in range(n):
            x = int(sigma[i])

            if win is None:
                j_min, j_max = 0, n - 1
            else:
                j_min = 0 if i - win < 0 else i - win
                j_max = (n - 1) if i + win > (n - 1) else i + win

            local_best_delta = NEG_INF
            local_best_j = -1

            # ---- LEFT moves: j in [j_min .. i-1]
            if j_min < i:
                seg = sigma[j_min:i]                  # elements crossed if moving x to the left
                vals = A[x, seg]                      # A[x, seg[k]]
                suffix = np.cumsum(vals[::-1])[::-1]  # suffix sums => sum_{t=j..i-1} A[x, sigma[t]]
                js = np.arange(j_min, i, dtype=np.int32)
                deltas = suffix                        # delta for moving to each j

                # Tabu filtering + aspiration
                expires = tabu_expire[x, js]
                is_tabu = it < expires
                cand_f = cur_f + deltas
                allowed = (~is_tabu) | (cand_f > best_f)

                if np.any(allowed):
                    masked = np.where(allowed, deltas, NEG_INF)
                    idx = int(np.argmax(masked))
                    d = int(masked[idx])
                    if d > local_best_delta:
                        local_best_delta = d
                        local_best_j = int(js[idx])

            # ---- RIGHT moves: j in [i+1 .. j_max]
            if i < j_max:
                seg = sigma[i + 1 : j_max + 1]        # elements crossed if moving x to the right
                vals = A[x, seg]
                prefix = np.cumsum(vals)              # sum_{t=i+1..j} A[x, sigma[t]]
                js = np.arange(i + 1, j_max + 1, dtype=np.int32)
                deltas = -prefix                      # delta = - sum(A[x, crossed])

                expires = tabu_expire[x, js]
                is_tabu = it < expires
                cand_f = cur_f + deltas
                allowed = (~is_tabu) | (cand_f > best_f)

                if np.any(allowed):
                    masked = np.where(allowed, deltas, NEG_INF)
                    idx = int(np.argmax(masked))
                    d = int(masked[idx])
                    if d > local_best_delta:
                        local_best_delta = d
                        local_best_j = int(js[idx])

            # Update global best move
            if local_best_j != -1 and local_best_delta > best_delta:
                best_delta = local_best_delta
                best_i = i
                best_j = local_best_j
                best_x = x

        # No move found (rare)
        if best_i == -1:
            break

        # Apply best move
        _apply_insert_inplace_np(sigma, best_i, best_j)
        cur_f = int(cur_f + best_delta)

        # Update tabu: forbid placing best_x again into position best_j for tenure
        tabu_expire[best_x, best_j] = it + t

        # Best-so-far update
        if cur_f > best_f:
            best_f = cur_f
            best_sigma = sigma.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break

    print("Iterations: ", it)

    elapsed = time.perf_counter() - start
    return best_sigma.tolist(), int(best_f), elapsed
import numpy as np
import time


def objective_function(W, sigma):
    W_reordered = W[np.ix_(sigma, sigma)]
    return int(np.sum(np.triu(W_reordered, k=1)))


def _apply_insert_inplace_np(sigma: np.ndarray, i: int, j: int) -> int:
    """Move sigma[i] to position j in-place; returns moved element."""
    x = int(sigma[i])
    if j > i:
        sigma[i:j] = sigma[i + 1 : j + 1]   # shift left
        sigma[j] = x
    else:
        sigma[j + 1 : i + 1] = sigma[j:i]   # shift right
        sigma[j] = x
    return x


def local_search_insert_first_improvement_simple_fast(W: np.ndarray, sigma0, max_iters: int = 10_000):
    """
    First-improvement local search for LOP with INSERT moves (optimized).
    - Scan i from 0..n-1
    - For each i try all j != i
    - Apply the first move with positive delta and restart scanning
    - Stop if a full scan finds no improvement or max_iters reached

    Returns: best_sigma, best_f, elapsed_time
    """
    start = time.perf_counter()

    W = np.asarray(W)
    n = W.shape[0]
    if W.shape[1] != n:
        raise ValueError("W must be square NxN")

    # Antisymmetric matrix for fast delta updates
    A = W - W.T

    sigma = np.array(list(sigma0), dtype=np.int32, copy=True)
    cur_f = objective_function(W, sigma.tolist())

    it = 0
    while it < max_iters:
        it += 1
        improved = False

        for i in range(n):
            x = int(sigma[i])

            # Try moving x to the LEFT (j = i-1 .. 0): delta accumulates as we extend the crossed segment
            d = 0
            for j in range(i - 1, -1, -1):
                y = int(sigma[j])
                d += int(A[x, y])          # delta for moving x to current j
                if d > 0:
                    _apply_insert_inplace_np(sigma, i, j)
                    cur_f += d
                    improved = True
                    break

            if improved:
                break

            # Try moving x to the RIGHT (j = i+1 .. n-1)
            d = 0
            for j in range(i + 1, n):
                y = int(sigma[j])
                d -= int(A[x, y])          # delta for moving x to current j
                if d > 0:
                    _apply_insert_inplace_np(sigma, i, j)
                    cur_f += d
                    improved = True
                    break

            if improved:
                break

        # No improving move found in a full scan => local optimum
        if not improved:
            break

    print("Iterations: ", it)
    
    elapsed = time.perf_counter() - start
    return sigma.tolist(), int(cur_f), elapsed

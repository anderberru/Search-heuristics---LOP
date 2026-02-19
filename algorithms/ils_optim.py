import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *

# Calculate delta to choose best neighbour (calculating objective_function is not optimized)
def insert_delta(W, perm, i, j):
    if i == j:
        return 0
    x = perm[i]
    if j > i:
        seg = perm[i + 1 : j + 1]
        return int(W[seg, x].sum() - W[x, seg].sum())
    else:
        seg = perm[j:i]
        return int(W[x, seg].sum() - W[seg, x].sum())

# Insert (optimizado con numpy)
def apply_insert_inplace(perm, i, j):
    if i == j:
        return
    x = perm[i]
    if j > i:
        perm[i:j] = perm[i + 1 : j + 1]
        perm[j] = x
    else:  # j < i
        perm[j + 1 : i + 1] = perm[j:i]
        perm[j] = x

# Perturbation
def perturbated_insert_np(sigma, strength, rng):
    n = sigma.shape[0]
    for _ in range(strength):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i != j:
            apply_insert_inplace(sigma, i, j) # Optimizado con numpy

# Local search (find local optimum)
def local_search_insert(W, perm, curr_val):
    """
    Local Search (canonical component): improve until local optimum using INSERT neighborhood.
    Best-improvement per step (scan all moves, apply best positive delta).
    """
    n = perm.shape[0]

    while True:
        best_delta = 0
        best_move = None

        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                d = insert_delta(W, perm, i, j)
                if d > best_delta:
                    best_delta = d
                    best_move = (i, j)

        if best_move is None:  # no improving move
            break

        i, j = best_move
        apply_insert_inplace(perm, i, j)
        curr_val += best_delta

    return perm, curr_val


def iterated_local_search_optim(
    W,
    sigma=None,
    perturbation_strength=None,
    max_iters=None,
    seed=0,
):
    """
    Iterated Local Search
    """
    start = time.perf_counter()

    W = np.asarray(W)
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    # Parameters
    if sigma is None:
        sigma = rng.permutation(n)
    if max_iters is None:
        max_iters = 100*n
    if perturbation_strength is None:
        perturbation_strength = max(2, int(0.1 * n))

    curr = np.asarray(sigma, dtype=int).copy()

    # Initial local optimum with local search
    curr_val = int(objective_function(W, curr))
    curr, curr_val = local_search_insert(W, curr, curr_val)

    best = curr.copy()
    best_val = curr_val

    for it in range(1, max_iters + 1):
        # Perturbation
        cand = curr.copy()
        perturbated_insert_np(cand, perturbation_strength, rng)

        # Local search applied to perturbed solution
        cand_val = int(objective_function(W, cand))
        cand, cand_val = local_search_insert(W, cand, cand_val)

        # Accept candidate as new current
        curr = cand
        curr_val = cand_val

        # track global best
        if cand_val > best_val:
            best_val = cand_val
            best = cand.copy()

        print("Iterations: ", it, end="\r")

    print()
    elapsed = time.perf_counter() - start
    return best.tolist(), int(best_val), elapsed

import numpy as np
import time
import random

import sys, os
sys.path.append(os.path.abspath(".."))
from functions import *


def insert_delta(W: np.ndarray, perm: np.ndarray, i: int, j: int) -> int:
    if i == j:
        return 0
    x = perm[i]
    if j > i:
        seg = perm[i + 1 : j + 1]
        return int(W[seg, x].sum() - W[x, seg].sum())
    else:
        seg = perm[j:i]
        return int(W[x, seg].sum() - W[seg, x].sum())


def apply_insert_inplace(perm: np.ndarray, i: int, j: int) -> None:
    if i == j:
        return
    x = perm[i]
    if j > i:
        perm[i:j] = perm[i + 1 : j + 1]
        perm[j] = x
    else:  # j < i
        perm[j + 1 : i + 1] = perm[j:i]
        perm[j] = x


def tabu_search_insert_optim(
    W,
    start_perm = None,
    max_iters = None,
    tenure_min = None, # minimum tabu tenure (prevents immediate cycling)
    tenure_max = None, # maximum tabu tenure (adds diversification)
    no_improve_limit = None, # stops search if best solution is not improved
    seed = 0,
):
    """
    Tabu Search for the Linear Ordering Problem using INSERT neighborhood.

    Tabu attribute used: (moved_element, destination_position)
    Aspiration: allow tabu move if it improves the global best.

    Returns:
      best_perm, best_value, info
    """
    start = time.perf_counter()
    
    W = np.asarray(W)
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    # Parameters
    if start_perm is None:
        start_perm = rng.permutation(n)
    if max_iters is None:
        max_iters = 100*n
    if tenure_min is None:
        tenure_min = max(5, int(0.2 * n))
    if tenure_max is None:
        tenure_max = max(tenure_min + 1, int(0.4 * n))
    if no_improve_limit is None:
        no_improve_limit = 30 * n

    perm = np.asarray(start_perm, dtype=int).copy()
    curr_val = objective_function(W, perm)
    best_perm = perm.copy()
    best_val = curr_val

    # tabu[(element, dest_pos)] = iteration_when_expires (strictly greater => tabu)
    tabu = {}

    iters_since_improve = 0

    for it in range(1, max_iters + 1):
        best_move = None  # (i, j, delta, new_val, element)
        best_candidate_val = -10**18

        # Search best admissible move (best-improvement)
        for i in range(n):
            x = int(perm[i])
            for j in range(n):
                if j == i:
                    continue

                delta = insert_delta(W, perm, i, j)
                cand_val = curr_val + delta

                attr = (x, j)
                is_tabu = (attr in tabu) and (tabu[attr] > it)

                # Aspiration: override tabu if it improves global best
                if is_tabu and cand_val <= best_val:
                    continue

                if cand_val > best_candidate_val:
                    best_candidate_val = cand_val
                    best_move = (i, j, delta, cand_val, x)

        if best_move is None:
            break  # no admissible move (rare unless tenure is extreme)

        i, j, delta, new_val, x = best_move

        # Apply move
        apply_insert_inplace(perm, i, j)
        curr_val = new_val

        # Update tabu tenure for the performed move
        tenure = int(rng.integers(tenure_min, tenure_max + 1))
        tabu[(int(x), int(j))] = it + tenure

        # Track global best
        if curr_val > best_val:
            best_val = curr_val
            best_perm = perm.copy()
            iters_since_improve = 0
        else:
            iters_since_improve += 1
            if iters_since_improve >= no_improve_limit:
                break
        
        print("Iterations: ", it, end="\r")

    print()

    elapsed = time.perf_counter() - start
    return best_perm.tolist(), int(best_val), elapsed

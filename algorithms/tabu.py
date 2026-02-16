import numpy as np
import time

import sys, os
sys.path.append(os.path.abspath(".."))
from functions import *

def apply_insert(perm, i, j):
    perm = perm.copy()
    x = perm[i]
    if j > i:
        perm[i:j] = perm[i+1:j+1]
        perm[j] = x
    else:
        perm[j+1:i+1] = perm[j:i]
        perm[j] = x
    return perm

def tabu_search_insert(
    W,
    start_perm = None,
    max_iters = None,
    tenure = None,
    seed = 0,
):
    
    start = time.perf_counter()

    W = np.asarray(W)
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    # Parameters
    if start_perm is None:
        start_perm = rng.permutation(n)
    if max_iters is None:
        max_iters = 100*n
    if tenure is None:
        tenure = int(0.3 * n)

    perm = np.asarray(start_perm, dtype=int).copy()
    curr = objective_function(W, perm)
    best_perm, best_val = perm.copy(), curr
    tabu = {}  # (element, new_pos) -> expire_iter

    for it in range(1, max_iters+1):
        best_move = None
        best_cand = -10**18

        for i in range(n):
            x = int(perm[i])
            for j in range(n):
                if j == i:
                    continue

                cand_perm = apply_insert(perm, i, j)
                cand_val = objective_function(W, cand_perm)

                is_tabu = (x, j) in tabu and tabu[(x, j)] > it
                if is_tabu and cand_val <= best_val:  # aspiration
                    continue

                if cand_val > best_cand:
                    best_cand = cand_val
                    best_move = (i, j, x, cand_perm)

        if best_move is None:
            break

        i, j, x, perm = best_move
        curr = best_cand
        tabu[(x, j)] = it + tenure

        if curr > best_val:
            best_val = curr
            best_perm = perm.copy()

    print("Iterations: ", it)

    elapsed = time.perf_counter() - start
    return best_perm, best_val, elapsed

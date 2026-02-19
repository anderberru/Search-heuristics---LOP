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


def _update_elite(elite, perm, val, elite_size):
    """
    Keep a list of up to elite_size best (val, perm) pairs.
    """
    elite.append((val, perm.copy()))
    elite.sort(key=lambda t: t[0], reverse=True)
    if len(elite) > elite_size:
        elite.pop()


def _elite_consensus_perm(elite, n, fallback_perm):
    """
    Build an intensification permutation from elite set:
    sort elements by their average position across elite solutions.
    """
    if not elite:
        return fallback_perm.copy()

    pos_sum = np.zeros(n, dtype=float)
    for _, p in elite:
        inv = np.empty(n, dtype=int)
        inv[p] = np.arange(n)
        pos_sum += inv

    avg_pos = pos_sum / len(elite)
    # Stable-ish tie-break using fallback order
    order = np.argsort(avg_pos, kind="mergesort")
    return order.astype(int)


def tabu_search_insert(
    W,
    start_perm=None,
    max_iters=None,
    tenure=None,
    seed=0,

    # NEW: choose memory modes
    medium_term=False,          # if True, uses elite-set intensification
    long_term=False,            # if True, uses frequency-based diversification

    # NEW: medium-term parameters
    elite_size=None,            # number of elite solutions to keep
    intensify_after=None,       # restart after this many non-improving iterations

    # NEW: long-term parameters
    long_lambda=None,           # penalty weight for frequently-used moves
):
    start = time.perf_counter()

    W = np.asarray(W)
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    # Basic parameters
    if start_perm is None:
        start_perm = rng.permutation(n)
    if max_iters is None:
        max_iters = 100 * n
    if tenure is None:
        tenure = int(0.1 * n)

    # Medium-term defaults
    if elite_size is None:
        elite_size = max(5, n // 10)  # simple default
    if intensify_after is None:
        intensify_after = 20 * n      # stagnation threshold

    # Long-term defaults
    if long_lambda is None:
        # Small, instance-scaled penalty (keeps it from dominating the objective)
        long_lambda = 0.01 * float(np.mean(np.abs(W)))

    perm = np.asarray(start_perm, dtype=int).copy()
    curr = objective_function(W, perm)
    best_perm, best_val = perm.copy(), curr

    # Short-term tabu list: (element, new_pos) -> expire_iter
    tabu = {}

    # Medium-term: store elite solutions
    elite = []
    if medium_term:
        _update_elite(elite, perm, curr, elite_size)

    # Long-term: frequency of move attributes (x, j)
    freq = np.zeros((n, n), dtype=np.int32) if long_term else None

    no_improve = 0

    for it in range(1, max_iters + 1):
        best_move = None
        best_score = -10**18   # score used for choosing the move (may include penalty)
        best_cand_val = -10**18  # true objective value (no penalties)

        for i in range(n):
            x = int(perm[i])
            for j in range(n):
                if j == i:
                    continue

                cand_perm = apply_insert(perm, i, j)
                cand_val = objective_function(W, cand_perm)

                # Tabu check (short-term memory)
                is_tabu = (x, j) in tabu and tabu[(x, j)] > it

                # Aspiration (always based on TRUE objective value)
                if is_tabu and cand_val <= best_val:
                    continue

                # Long-term diversification: penalize frequently used move attributes
                if long_term:
                    score = cand_val - long_lambda * freq[x, j]
                else:
                    score = cand_val

                if score > best_score:
                    best_score = score
                    best_cand_val = cand_val
                    best_move = (i, j, x, cand_perm)

        if best_move is None:
            break

        i, j, x, perm = best_move
        curr = best_cand_val

        # Update short-term tabu
        tabu[(x, j)] = it + tenure

        # Update long-term frequency memory
        if long_term:
            freq[x, j] += 1

        # Update best and elite
        if curr > best_val:
            best_val = curr
            best_perm = perm.copy()
            no_improve = 0
        else:
            no_improve += 1

        if medium_term:
            _update_elite(elite, perm, curr, elite_size)

        # Medium-term intensification trigger: restart from elite consensus
        if medium_term and no_improve >= intensify_after:
            perm = _elite_consensus_perm(elite, n, best_perm)
            curr = objective_function(W, perm)
            tabu.clear()          # common choice after restart
            no_improve = 0

    print("Iterations: ", it)
    elapsed = time.perf_counter() - start
    return best_perm, best_val, elapsed

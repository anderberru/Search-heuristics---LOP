import numpy as np
import time

def objective_function(W, sigma):
    """
    Calculate the objective function of Linear Ordering Problem. Reorder the weights matrix and sum the upper triangle of the weights matrix.
    
    Parameters
    ----------
    W : matrix of int
        Matrix of weights
    sigma : list of int
        Permutation to reorder matrix of weights
    
    Returns
    -------
    int
        Sum of the upper triangle of the reordered matrix
    """
    # Version 1: basic method
    # f = 0
    # for i in range(W.shape[0]):
    #     for j in range(i+1, W.shape[1]):
    #         f += W[sigma[i], sigma[j]]
    # return f

    # Version 2: np method, more optimal
    # ix_: Return the cross-section of the array W at the specified indices.
    W_reordered = W[np.ix_(sigma, sigma)] # Reorder the matrix W according to the permutation sigma
    # triu: convers every element below the diagonal to zero (wincluding the diagonal, with k=1)
    f = np.sum(np.triu(W_reordered, k=1)) # sum of the upper triangle
    return f


def load_matrix_from_file(filepath, dtype=int):
    """
    Loads a square NumPy matrix from a text file.
    
    File format:
    - First line: integer N (matrix size)
    - Next N lines: N space- or tab-separated values per line
    
    Parameters
    ----------
    filepath : str
        Path to the file
    dtype : data-type, optional
        Desired NumPy data type (default: int)
    
    Returns
    -------
    np.ndarray
        NxN NumPy array
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Read matrix size
    n = int(lines[0].strip())

    # Read matrix values
    data = []
    for line in lines[1:n+1]:
        row = list(map(dtype, line.split()))
        if len(row) != n:
            raise ValueError("Row length does not match matrix size")
        data.append(row)

    return np.array(data, dtype=dtype)

# def neighbour_insert(sigma, i, j):
#     neighbour = list(sigma.copy())
#     element = neighbour.pop(i)  # Remove the element at index i
#     neighbour.insert(j, element)  # Insert it at index j
#     return neighbour

def N_insert(sigma):
    N = []
    n = len(sigma)

    for i in range(n):
        base = list(sigma)
        element = base.pop(i)

        for j in range(n):
            if j != i:
                neighbour = base.copy()
                neighbour.insert(j, element)
                N.append(neighbour)
    return N

def local_search_insert_bestFirst(W, sigma, max_iters=1000):
    start_timer = time.perf_counter()
    best_sigma = list(sigma)
    best_f = objective_function(W, sigma)
    visited = set()
    iters = 0


    n = len(sigma)
    stuck = False
    new_best_found = False
    n = len(sigma)

    while not stuck and iters < max_iters:
        # N_instert
        # print(f"Iteration {iters}, best_f: {best_f}")
        visited = set()
        new_best_found = False
        for i in range(n):
            base = list(best_sigma)
            element = base.pop(i)

            for j in range(n):
                if j != i:
                    neighbour = base.copy()
                    neighbour.insert(j, element)
                    
                    if tuple(neighbour) not in visited:
                        visited.add(tuple(neighbour))  # Mark this neighbour as visited
                        # compare objective function value of neighbour with best_f
                        f_neighbour = objective_function(W, neighbour)
                        
                        if f_neighbour > best_f:
                            best_f = f_neighbour
                            best_sigma = neighbour
                            new_best_found = True
                            break
            if new_best_found:
                break
        if not new_best_found:
            stuck = True
        iters += 1

    end_timer = time.perf_counter()
    elapsed_time = end_timer - start_timer
    print("Total iterations: " + str(iters))
    return best_sigma, best_f, elapsed_time


## ------------------------------------------------------------------------------ ##
import time
from typing import List, Tuple, Optional


def lop_objective(W: List[List[float]], perm: List[int]) -> float:
    n = len(perm)
    s = 0.0
    for p in range(n):
        a = perm[p]
        row = W[a]
        for q in range(p + 1, n):
            s += row[perm[q]]
    return s


def insert_move_inplace(perm: List[int], i: int, j: int) -> None:
    """Move element at position i to position j (after removal)."""
    x = perm.pop(i)
    perm.insert(j, x)


def insert_delta(W: List[List[float]], perm: List[int], i: int, j: int) -> float:
    """
    Delta in LOP objective if we remove perm[i] and insert it at position j.
    Runs in O(n) by only considering pairs whose relative order changes.

    Works for any i != j. 'j' is the index in the permutation AFTER removal (Python list insert semantics).
    """
    if i == j:
        return 0.0

    n = len(perm)
    x = perm[i]

    # If moving forward (to a larger index), after removal the target index decreases by 1
    if j > i:
        # element passes over positions i+1 ... j
        # Before: x before each y in (i+1..j) contributes W[x][y]
        # After:  each y before x contributes W[y][x]
        delta = 0.0
        for k in range(i + 1, j + 1):
            y = perm[k]
            delta += W[y][x] - W[x][y]
        return delta
    else:
        # moving backward (to a smaller index), element passes over positions j ... i-1
        # Before: each y in (j..i-1) before x contributes W[y][x]
        # After:  x before y contributes W[x][y]
        delta = 0.0
        for k in range(j, i):
            y = perm[k]
            delta += W[x][y] - W[y][x]
        return delta


def local_search_insert_first_improvement(
    W: List[List[float]],
    sigma: List[int],
    max_iters: int = 1000,
) -> Tuple[List[int], float, float, int, int]:
    """
    First-improvement local search with insertion moves for LOP.
    Returns (best_perm, best_value, elapsed_time, iterations, moves).
    """
    start = time.perf_counter()

    best_sigma = list(sigma)
    best_f = lop_objective(W, best_sigma)

    n = len(best_sigma)
    iters = 0
    moves = 0

    while iters < max_iters:
        improved = False

        for i in range(n):
            for j in range(n):
                if j == i:
                    continue

                d = insert_delta(W, best_sigma, i, j)
                if d > 0:
                    # Apply move
                    insert_move_inplace(best_sigma, i, j)
                    best_f += d
                    moves += 1
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

        iters += 1

    print("Total iterations: " + str(iters))

    elapsed = time.perf_counter() - start
    return best_sigma, best_f, elapsed
## ---------------------------------------------------------------------------------- ##


def local_search_insert_greedy(W, sigma, max_iters=1000):
    start_timer = time.perf_counter()
    best_sigma = sigma
    best_f = objective_function(W, sigma)
    visited = set()

    n = len(sigma)
    stuck = False
    new_best_found = False
    n = len(sigma)
    iters = 0

    while not stuck and iters < max_iters:
        # N_instert
        visited = set()
        new_best_found = False
        for i in range(n):
            base = list(best_sigma)
            element = base.pop(i)

            for j in range(n):
                if j != i:
                    neighbour = base.copy()
                    neighbour.insert(j, element)
                    
                    if tuple(neighbour) not in visited:
                        visited.add(tuple(neighbour))  # Mark this neighbour as visited
                        # compare objective function value of neighbour with best_f
                        f_neighbour = objective_function(W, neighbour)

                        if f_neighbour > best_f:
                            best_f = f_neighbour
                            best_sigma = neighbour
                            new_best_found = True
                        
        if not new_best_found:
            stuck = True

        iters += 1

    end_timer = time.perf_counter()
    elapsed_time = end_timer - start_timer
    print("Total iterations: " + str(iters))
    return best_sigma, best_f, elapsed_time

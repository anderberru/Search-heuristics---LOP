import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *

def iterated_local_search(W, sigma=None, perturbation_strength=2, max_iters=1000, seed=0):
    start_timer = time.perf_counter()
    n = W.shape[0]
    rng = np.random.default_rng(seed)

    # Parameters
    if sigma is None:
        sigma = rng.permutation(n)

    best_sigma = sigma
    best_f = objective_function(W, sigma)
    visited = set()
    
    stuck = False
    new_best_found = False
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
            # Perturbation
            best_sigma = perturbated_insert(best_sigma, perturbation_strength)

        iters += 1
        print("Iterations: ", iters, end="\r")

    end_timer = time.perf_counter()
    elapsed_time = end_timer - start_timer
    return best_sigma, best_f, elapsed_time

def perturbated_insert(sigma, strength):
    sigma = list(sigma)
    n = len(sigma)

    for _ in range(strength):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i != j:
            element = sigma.pop(i)
            sigma.insert(j, element)

    return sigma



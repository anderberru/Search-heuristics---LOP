import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *

def iterated_local_search(W, sigma, perturbation_strength=2, max_iters=1000):
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




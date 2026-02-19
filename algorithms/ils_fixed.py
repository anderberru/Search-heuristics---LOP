import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *
import random
import time

def local_search_insert(W, sigma):
    """
    MODIFIED: canonical LocalSearch operator.
    Runs until no improving INSERT neighbor exists.
    This is a standard first-improvement local search (restart after each improvement).
    """
    n = len(sigma)
    sigma = list(sigma)
    curr_f = objective_function(W, sigma)

    improved = True
    while improved:
        improved = False

        # Explore neighborhood of *current* sigma
        for i in range(n):
            base = sigma.copy()
            element = base.pop(i)

            for j in range(n):
                if j == i:
                    continue

                neighbour = base.copy()
                neighbour.insert(j, element)

                f_neighbour = objective_function(W, neighbour)
                if f_neighbour > curr_f:
                    # Accept improvement and RESTART neighborhood exploration
                    sigma = neighbour
                    curr_f = f_neighbour
                    improved = True
                    break
            if improved:
                break

    return sigma, curr_f

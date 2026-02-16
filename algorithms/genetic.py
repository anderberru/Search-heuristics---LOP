import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *

def choose_parents(population):
    return 0, 0

def cross_parents(parent1, parent2):
    return 0, 0

def mutate_child(child):
    return 0

def choose_new_population(population):
    return []


def genetic_algorithm(W, population0, generations=1000):
    start_timer = time.perf_counter()
    k = 0
    n = len(population0)
    pk = population0.copy()

    while k < generations:
        pk_prime = pk.copy()
        k += 1
        for _ in range(n//2):
            # choose parents based on a criteria
            parent1, parent2 = choose_parents(pk)

            # cross parents
            child1, child2 = cross_parents(parent1, parent2)

            # mutate new children
            child1_mutated = mutate_child(child1)
            child2_mutated = mutate_child(child2)

            # introduce new individuals to the population
            pk_prime.extend([child1_mutated, child2_mutated])

        pk = choose_new_population(pk_prime)

    # take the best individual from the final population
    best_individual = max(pk, key=lambda ind: objective_function(W, ind))
    best_f = objective_function(W, best_individual)

    end_timer = time.perf_counter()
    elapsed_time = end_timer - start_timer
    return best_individual, best_f, elapsed_time






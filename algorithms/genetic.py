import sys
import os
sys.path.append(os.path.abspath(".."))
from functions import *

def choose_parents(population, population_size, W, method="proportional"):
    if method == "proportional":
        # selection function proportional to objective function values
        sum_f = sum(objective_function(W, ind) for ind in population)

        # Calculate selection probabilities for each individual
        probabilities = [objective_function(W, ind) / sum_f for ind in population]

        # Randomly select two parents based on the calculated probabilities
        selected_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
        parent1 = population[selected_indices[0]]
        parent2 = population[selected_indices[1]]
    return parent1, parent2

def cross_parents(parent1, parent2, method="order_crossover"):
    child1, child2 = [], []
    if method == "order_crossover":
        n = len(parent1)
        i, j = sorted(np.random.choice(n, 2, replace=False))

        # initialize children with -1 to indicate unfilled positions
        child1 = [-1] * n
        child2 = [-1] * n

        # Copy the segment from parent1 to child1 and from parent2 to child2
        child1[i:j] = parent1[i:j]
        child2[i:j] = parent2[i:j]

        # fill the remaining positions in child1 with the order of parent2
        remaining1 = [x for x in parent2 if x not in child1]
        idx = 0
        for k in range(n):
            if child1[k] == -1:
                child1[k] = remaining1[idx]
                idx += 1

        # fill the remaining positions in child2 with the order of parent1
        remaining2 = [x for x in parent1 if x not in child2]
        idx = 0
        for k in range(n):
            if child2[k] == -1:
                child2[k] = remaining2[idx]
                idx += 1

    return child1, child2

def mutate_child(child, method="swap"):
    mutated_child = child.copy()
    if method == "swap":
        n = len(mutated_child)
        i, j = np.random.choice(n, size=2, replace=False)
        mutated_child[i], mutated_child[j] = mutated_child[j], mutated_child[i]
    return mutated_child

def choose_new_population(population_prime, population_size, W, method="elitist"):
    new_population = []
    if method == "elitist":
        # keep the best individuals from the current population
        sorted_population = sorted(population_prime, key=lambda ind: objective_function(W, ind), reverse=True)
        # keep the best individuals that fit in the population size
        new_population = sorted_population[:population_size]
    return new_population

    


def genetic_algorithm(W, population0, generations=1000, parent_selection_method="proportional", crossover_method="order_crossover", mutation_method="swap", new_population_method="elitist"):
    start_timer = time.perf_counter()
    k = 0
    n = len(population0)
    pk = population0.copy()

    while k < generations:
        pk_prime = pk.copy()
        k += 1
        for _ in range(n//2):
            # choose parents based on a criteria
            parent1, parent2 = choose_parents(pk, n, W, method=parent_selection_method)

            # cross parents
            child1, child2 = cross_parents(parent1, parent2, method=crossover_method)

            # mutate new children
            child1_mutated = mutate_child(child1, method=mutation_method)
            child2_mutated = mutate_child(child2, method=mutation_method)

            # introduce new individuals to the population
            pk_prime.extend([child1_mutated, child2_mutated])

        pk = choose_new_population(pk_prime, n, W, method=new_population_method)

    # take the best individual from the final population
    best_individual = max(pk, key=lambda ind: objective_function(W, ind))
    best_f = objective_function(W, best_individual)

    end_timer = time.perf_counter()
    elapsed_time = end_timer - start_timer
    return best_individual, best_f, elapsed_time






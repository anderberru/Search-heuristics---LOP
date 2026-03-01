from functions import *
from algorithms.genetic import *


# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
# A = load_matrix_from_file("instances/Cebe.lop.n30.4")
A = load_matrix_from_file("instances/N-r100a2")
# A = load_matrix_from_file("instances/N-r250e0")
# A = load_matrix_from_file("instances/mini")
# print(A)
# print(A.shape)
# print(type(A))


population_size = 150
# random initial population of permutations
population0 = [np.random.permutation(A.shape[0]) for _ in range(population_size)]
initial_max = max(objective_function(A, ind) for ind in population0)
print("Initial population max objective function value: " + str(initial_max))
best_sigma, best_f, elapsed_time = genetic_algorithm(A, population0, generations=400, 
                                                     parent_selection_method="tournament", tournament_size=2, 
                                                     crossover_method="order_crossover", 
                                                     mutation_method="insert", 
                                                     new_population_method="elitist_with_immigrants")

delta = best_f - initial_max
print(f"Improvement over initial population: {delta:.4f}")
delta_relative = (best_f - initial_max) / abs(initial_max) if initial_max != 0 else float('inf')
print(f"Relative improvement over initial population: {delta_relative:.2f}")
print(f"Genetic algorithm completed in {elapsed_time:.4f} seconds.")
# print("best sigma: " + str(np.array(best_sigma)))
print("best objective function value: " + str(best_f))

# from plyer import notification
# notification.notify(message='Execution Finished!')

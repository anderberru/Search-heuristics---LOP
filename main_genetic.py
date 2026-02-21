from functions import *
from algorithms.genetic import *


# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
# A = load_matrix_from_file("instances/Cebe.lop.n30.4")
A = load_matrix_from_file("instances/N-r100a2")
# A = load_matrix_from_file("instances/mini")
# print(A)
# print(A.shape)
# print(type(A))


population_size = 20
# random initial population of permutations
population0 = [np.random.permutation(A.shape[0]) for _ in range(population_size)]
best_sigma, best_f, elapsed_time = genetic_algorithm(A, population0, generations=1000, 
                                                     parent_selection_method="proportional", 
                                                     crossover_method="order_crossover", 
                                                     mutation_method="swap", 
                                                     new_population_method="elitist")

print(f"Genetic algorithm completed in {elapsed_time:.4f} seconds.")
# print("best sigma: " + str(np.array(best_sigma)))
print("best objective function value: " + str(best_f))

# from plyer import notification
# notification.notify(message='Execution Finished!')

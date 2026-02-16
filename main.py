from functions import *
from algorithms.ils import *


# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
# A = load_matrix_from_file("instances/Cebe.lop.n30.4")
A = load_matrix_from_file("instances/N-r100a2")
# A = load_matrix_from_file("instances/mini")
# print(A)
# print(A.shape)
# print(type(A))

rng = np.random.default_rng(42)
sigma = np.random.permutation(A.shape[0])
f = objective_function(A, sigma)
print("original sigma: " + str(sigma))
print("initial objective function value: " + str(f))
# N =N_insert(sigma)
# print(np.array(N))
# print("length N: "+ str(len(N)))
# print("n(n-1): " + str(len(sigma)*(len(sigma)-1)))

# best_sigma, best_f, elapsed_time = local_search_insert_greedy(A, sigma, max_iters=1000)
# print(f"Local search (insert) completed in {elapsed_time:.4f} seconds.")
# print("best sigma: " + str(np.array(best_sigma)))
# print("best objective function value: " + str(best_f))


# best_sigma, best_f, elapsed_time = local_search_insert_first_improvement(A, sigma, max_iters=1000)
best_sigma, best_f, elapsed_time = iterated_local_search(A, sigma, perturbation_strength=2, max_iters=1000)
print(f"Completed in {elapsed_time:.4f} seconds.")
# print("best sigma: " + str(np.array(best_sigma)))
print("best objective function value: " + str(best_f))
from plyer import notification
notification.notify(message='Execution Finished!')

# ils result:
# initial objective function value: 81074
# Total iterations: 1000
# Local search (insert) completed in 1055.0506 seconds.
# best objective function value: 145228
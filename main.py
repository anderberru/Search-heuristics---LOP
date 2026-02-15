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


best_sigma, best_f, elapsed_time = local_search_insert_first_improvement(A, sigma, max_iters=1000)
print(f"Local search (insert) completed in {elapsed_time:.4f} seconds.")
print("best sigma: " + str(np.array(best_sigma)))
print("best objective function value: " + str(best_f))
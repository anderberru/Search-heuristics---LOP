from functions import *
from algorithms.first_improvement import *


seed = 42
rng = np.random.default_rng(seed)

# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
W = load_matrix_from_file("instances/Cebe.lop.n30.4")
# W = load_matrix_from_file("instances/N-r250e0")

best_sigma, best_f, t = local_search_insert_first_improvement_simple_fast(
    W, 
    rng.permutation(W.shape[0]),
    max_iters=1000
)

print("Best f:", best_f, "time:", t)
from functions import *
from algorithms.tabu import *


seed = 42
rng = np.random.default_rng(seed)

# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
# A = load_matrix_from_file("instances/Cebe.lop.n30.4")
W = load_matrix_from_file("instances/N-r250e0")

best_sigma, best_f, t = tabu_search_insert(
    W, 
    rng.permutation(W.shape[0]),
    max_iters=10000,
    max_no_improve=1500,
    window_k=None,
    seed=seed
)

print("Best f:", best_f, "time:", t)


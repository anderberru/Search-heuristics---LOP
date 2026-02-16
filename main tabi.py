from functions import *
from algorithms.tabu2 import *



# W = load_matrix_from_file("instances/Cebe.lop.n10.1")
W = load_matrix_from_file("instances/Cebe.lop.n30.4")
# W = load_matrix_from_file("instances/N-r100a2")

best_sigma, best_f, t = tabu_search_insert(
    W, 
    medium_term=True, 
    long_term=True)


print("Best f:", best_f, "time:", t)


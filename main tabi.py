from functions import *
from algorithms.ils_optim import *



# W = load_matrix_from_file("instances/Cebe.lop.n10.1")
# W = load_matrix_from_file("instances/Cebe.lop.n30.4")
# W = load_matrix_from_file("instances/N-r100a2")
W = load_matrix_from_file("instances/N-r250e0")

best_sigma, best_f, t = iterated_local_search_optim(
    W, 
    max_iters=1000,
    # medium_term=True, 
    # long_term=True
    )


print("Best f:", best_f, "time:", t)
from plyer import notification
notification.notify(message='Execution Finished!')

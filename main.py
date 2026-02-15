from functions import *



# A = load_matrix_from_file("instances/Cebe.lop.n10.1")
# A = load_matrix_from_file("instances/Cebe.lop.n30.4")
# A = load_matrix_from_file("instances/N-r250d0")
A = load_matrix_from_file("instances/mini")
# print(A)
# print(A.shape)
# print(type(A))

rng = np.random.default_rng(42)
sigma = rng.permutation(A.shape[0])
# f = objective_function(A, sigma)
# print(f)
print("original sigma: " + str(sigma))
N =N_insert(sigma)
print(np.array(N))
print("length N: "+ str(len(N)))
print("n(n-1): " + str(len(sigma)*(len(sigma)-1)))

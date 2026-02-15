import numpy as np

def objective_function(W, sigma):
    """
    Calculate the objective function of Linear Ordering Problem. Reorder the weights matrix and sum the upper triangle of the weights matrix.
    
    Parameters
    ----------
    W : matrix of int
        Matrix of weights
    sigma : list of int
        Permutation to reorder matrix of weights
    
    Returns
    -------
    int
        Sum of the upper triangle of the reordered matrix
    """
    # Version 1: basic method
    # f = 0
    # for i in range(W.shape[0]):
    #     for j in range(i+1, W.shape[1]):
    #         f += W[sigma[i], sigma[j]]
    # return f

    # Version 2: np method, more optimal
    # ix_: Return the cross-section of the array W at the specified indices.
    W_reordered = W[np.ix_(sigma, sigma)] # Reorder the matrix W according to the permutation sigma
    # triu: convers every element below the diagonal to zero (wincluding the diagonal, with k=1)
    f = np.sum(np.triu(W_reordered, k=1)) # sum of the upper triangle
    return f


def load_matrix_from_file(filepath, dtype=int):
    """
    Loads a square NumPy matrix from a text file.
    
    File format:
    - First line: integer N (matrix size)
    - Next N lines: N space- or tab-separated values per line
    
    Parameters
    ----------
    filepath : str
        Path to the file
    dtype : data-type, optional
        Desired NumPy data type (default: int)
    
    Returns
    -------
    np.ndarray
        NxN NumPy array
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Read matrix size
    n = int(lines[0].strip())

    # Read matrix values
    data = []
    for line in lines[1:n+1]:
        row = list(map(dtype, line.split()))
        if len(row) != n:
            raise ValueError("Row length does not match matrix size")
        data.append(row)

    return np.array(data, dtype=dtype)

# def neighbour_insert(sigma, i, j):
#     neighbour = list(sigma.copy())
#     element = neighbour.pop(i)  # Remove the element at index i
#     neighbour.insert(j, element)  # Insert it at index j
#     return neighbour

def N_insert(sigma):
    N = []
    n = len(sigma)

    for i in range(n):
        base = list(sigma)
        element = base.pop(i)

        for j in range(n):
            if j != i:
                neighbour = base.copy()
                neighbour.insert(j, element)
                N.append(neighbour)
    return N

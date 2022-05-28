"""
Power method of SVD algorithm.
"""
import numpy as np

epsilon = 1e-6
def svd_1d(matrix):
    """
    SVD for 1 rank (with the greatest singular value).
    """
    _, n = matrix.shape
    #initializing random vector
    eigenvector = np.random.normal(0, 1, (n))
    eigenvector = eigenvector/(sum(x * x for x in eigenvector))**(1/2)
    symmetric = matrix.T @ matrix
    while True:
        new_eigenvector = symmetric @ eigenvector
        new_eigenvector = new_eigenvector / (sum(x * x for x in new_eigenvector))**(1/2)
        if sum((x-y)**(2) for x, y in zip(eigenvector, new_eigenvector))**(1/2) < epsilon:
            break
        eigenvector = new_eigenvector
    return eigenvector

def svd(matrix, k):
    """
    Returns SVD for k ranks.
    """
    m, n = matrix.shape
    if k > min(m, n):
        raise ValueError("k should be <= number of rows and number of columns")
    computed_svd = []
    computation_matrix = matrix.copy()
    for i in range(k):
        if i != 0:
            computation_matrix -= computed_svd[i-1][0]*np.outer(computed_svd[i-1][1], (computed_svd[i-1][2]))
        v = svd_1d(computation_matrix)
        u = matrix@v
        sigma = (sum(x * x for x in u))**(1/2)
        u = u/sigma
        computed_svd.append((sigma, u, v))
    singular_values, us, vs = [np.array(x) for x in zip(*computed_svd)]
    return singular_values, us.T, vs

if __name__ == "__main__":
    matrix = np.array([[2, 0], [1, 2]], dtype='float64')
    result = svd(matrix, 2)
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    values, left, rigth = svd(matrix, 2)
    print(np.allclose(np.absolute(u), np.absolute(left)))
    print(np.allclose(np.absolute(s), np.absolute(values)))
    print(np.allclose(np.absolute(v), np.absolute(rigth)))


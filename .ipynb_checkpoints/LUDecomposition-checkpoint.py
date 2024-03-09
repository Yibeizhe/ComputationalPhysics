# Author: Junfei Ding, Guizhou University, Date: 2023-10-9
import numpy as np
def lu_decomposition(A):
    """
    Perform LU decomposition on matrix A using numpy slicing.
    
    Parameters:
    A (np.array): Square matrix of size (n, n).
    
    Returns:
    L (np.array): Lower triangular matrix of size (n, n).
    U (np.array): Upper triangular matrix of size (n, n).
    """
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A)
    
    for i in range(n):
        for j in range(i+1, n):
            coeff=U[j, i] / U[i, i]
            L[j, i] = coeff
            U[j, i:] -= coeff * U[i, i:]
    
    return L, U
if __name__=="__main__":
    A = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]], dtype=float)
    L, U = lu_decomposition(A)
    print("A matix:\n",A)
    print("Lower matix:\n",L)
    print("Upper matix:\n",U)

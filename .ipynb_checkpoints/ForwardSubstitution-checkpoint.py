# Author: Junfei Ding, Guizhou University, Date: 2023-10-9
import numpy as np
def forward_substitution(L, b):
    """
    Perform forward substitution to solve
    the lower triangular system Lx = b.
    
    Parameters:
    - L: Lower triangular matrix (2D numpy array)
    - b: Right-hand side vector (1D numpy array)
    
    Returns:
    - x: Solution vector (1D numpy array)
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    
    return x

if __name__=="__main__":
    L = np.array([[1, 0, 0], [2, 1, 0], [3, 4, 1]])
    b = np.array([1, 4, 7])
    x=forward_substitution(L,b)
    print(x)

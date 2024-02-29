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

def backward_substitution(U, b):
    """
    Perform backward substitution to solve 
    the upper triangular system Ux = b.
    
    Parameters:
    - U: Upper triangular matrix (2D numpy array)
    - b: Right-hand side vector (1D numpy array)
    
    Returns:
    - x: Solution vector (1D numpy array)
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x

# Author: Junfei Ding, Guizhou University, Date: 2023-10-9
import numpy as np
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
    for i in reversed(range(n)):
        x[i]=(b[i]-U[i, i+1:]@x[i+1:])/ U[i, i]
    return x
if __name__=="__main__":
    U = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    b = np.array([1, 2, 3])
    x=backward_substitution(U,b)
    print(x)

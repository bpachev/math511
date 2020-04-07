import numpy as np
from numpy.linalg import solve
from scipy.sparse import diags

def tridiag(low, mid, high, b):
    """
    Accepts an array low, mid, high with the lower, main, and upper diagonals.
    This is to ensure O(n) storage.
    b is the right hand side of a tridiagonal system
    """
    if mid.size-1 != low.size or high.size != low.size: raise ValueError("Incorrect diagonal lengths.")
    #Step 1: compute the Crout factorization A = LU
    l_main = np.zeros_like(mid)
    u_diag = np.zeros_like(high)
    l_main[0] = mid[0]
    n = mid.size
    for i in xrange(n-1):
        u_diag[i] = high[i] / l_main[i]
        l_main[i+1] = mid[i+1] - low[i]*u_diag[i]
    #Step 2: Use forward substitution to solve the linear system Ly = b
    y = np.zeros(n)
    y[0] = b[0]/l_main[0]
    for i in xrange(n-1):
        y[i+1] = (b[i+1] - low[i]*y[i]) / l_main[i+1]
    #Step 3: Use backwards substitution to solve Ux = y
    x = np.zeros(n)
    x[-1] = y[-1]
    for i in xrange(n-2,-1,-1):
        x[i] = y[i] - u_diag[i]*x[i+1]
    return x

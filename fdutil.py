import numpy as np
from numpy.linalg import solve
"""
A utility file containing routines for generating finite difference coefficients.
"""

def FDCoeffs(xb, x, k, ret_error = False):
	"""
	Input:
		xb -- a real number, the point at which to approximate the derivative
		x -- a vector of grid points. Must have length >= k+1
		k -- an integer, the order of derivative being approximated
		ret_error -- boolean
	Returns:
		c -- the coefficients of the finite difference
		error (optional) -- the coefficient by the next order derivative, or the truncation error
	"""

	#Construct the Vandermonde system
	n = len(x)
	A = np.zeros((n,n))
	b = np.zeros(n)
	b[k] = 1
	denom = 1.
	for j in range(n):
		A[j,:] = (x-xb)**j/denom
		denom *= (j+1)
	
	c = solve(A,b)
	err0 = np.dot(c,A[-1]*(x-xb)/n)
	err1 = np.dot(c,A[-1]*(x-xb)**2/n/(n+1))
	return c, err0,err1 if ret_error else c

def FDStencil(k, j, verbose=True):
	"""
		k -- the order of the derivative
		j -- a vector of integer indices on a uniformly spaced grid
	"""

	c, err0,err1 = FDCoeffs(0,j,k,True)
	if not verbose: return c
	print("Coefficients: h^{} * ".format(-k),c)
	#Centered difference
	if abs(err0) < 1e-14:
		print("Error: {:.6f}*h^{}*u(x)^({})".format(err1, len(j)-k+1, len(j)+1))
	else:
		print("Error: {:.6f}*h^{}*u(x)^({})".format(err0, len(j)-k, len(j)))

	return c


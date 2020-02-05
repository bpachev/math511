import numpy as np
from numpy.linalg import solve
"""
A utility file containing routines for generating finite difference coefficients.
"""

#Here is the MATLAB code from the book, translated to Python

def stable_fdcoeffs(xbar, x, k):
	n = len(x)
	if k >= n:
		error('*** length(x) must be larger than k')

	m = k;   # change to m=n-1 if you want to compute coefficients for all
	         # possible derivatives.  Then modify to output all of C.
	#Add padding so the 1-based indexing actually works
	x_new = np.zeros(len(x)+1)
	x_new[1:] = x
	x = x_new
	c1 = 1;
	c4 = x[1] - xbar
	#Add an extra row and column so that the 1-based indexing doesn't break
	C = np.zeros((n-1+1+1,m+1+1))
	C[1,1] = 1
	for i in range(1,n):
		i1 = i+1
		mn = min(i,m)
		c2 = 1
		c5 = c4
		c4 = x[i1] - xbar
		for j in range(0,i):
			j1 = j+1
			c3 = x[i1] - x[j1]
			c2 = c2*c3
			if j==i-1:
				for s in list(range(1,mn+1))[::-1]:
					s1 = s+1
					C[i1,s1] = c1*(s*C[i1-1,s1-1] - c5*C[i1-1,s1])/c2
				C[i1,1] = -c1*c5*C[i1-1,1]/c2
			for s in list(range(1,mn+1))[::-1]:
				s1 = s+1
				C[j1,s1] = (c4*C[j1,s1] - s*C[j1,s1-1])/c3
			C[j1,1] = c4*C[j1,1]/c3
		c1 = c2
	return C[1:,-1]

def FDCoeffs(xb, x, k, ret_error = False, stable=False):
	"""
	Input:
		xb -- a real number, the point at which to approximate the derivative
		x -- a vector of grid points. Must have length >= k+1
		k -- an integer, the order of derivative being approximated
		ret_error -- boolean
		stable -- whether to avoid underflow
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
		if not stable or j <= k: A[j,:] = (x-xb)**j/denom
		if stable and j > k:
			A[j] = A[j-1] * (x-xb)
			A[j] /= np.sqrt(np.sum(A[j]**2))
		denom *= (j+1)
	try:
		c = solve(A,b)
	except:
		print stable_fdcoeffs(xb, x, k), stable_fdcoeffs(1, np.arange(0,3), 2)
#		print A, A.shape, xb, x-xb
		raise
	err0 = np.dot(c,A[-1]*(x-xb)/n)
	err1 = np.dot(c,A[-1]*(x-xb)**2/n/(n+1))
	return (c, err0,err1) if ret_error else c

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


import numpy as np
import fdutil as fd
#Would be interesting to compare solve_banded and spdiags + spsolve
#from scipy.sparse import spdiags
from scipy.linalg import solve_banded

def bvp(f, n = 20, a=0,b=1,order=2, alpha=('D', 0), beta=('D',1), alpha_order=None, beta_order=None):
	"""
	A bvp solver for the BVP u'' = f(x).
	Parameters:
	f - the forcing function
	a,b - the endpoints
	alpha - the left boundary condition, a tuple
		The first entry in the tuple is either 'N' for Neumann boundary conditions ( or 'D' for Dirichlet)
	beta - the right boundary condition
	order - the order of accuracy to use in the discretization.
		If an odd order is passed, we silently up it to the next even number, as we always use a centered difference approximation.
	n - the number of subintervals
	alpha_order - the order of finite difference approximation to use for u'(a) if a Neumann boundary condition was used at x=a
	beta_order - same as alpha_order, but at x=b
	Returns:
	x - the approximated solution
	"""
	#Ensure the order is even	
	order += order % 2
	half_order = int(order/2)
	center_coeffs = fd.FDStencil(2, np.arange(-half_order, half_order+1), verbose=False)
	left_coeffs = []
	right_coeffs = []
	print(center_coeffs)
	res = np.zeros(n+1)
	if alpha[0] == 'N':
		#we need to add an equation at the left, as u(a) is unknown
		if alpha_order is None: alpha_order = order
		alpha_n = 1
		pass
	else:
		alpha_n = 0
		res[0] = alpha[1]
	if beta[0] == 'N':
		#We need to add an equation at the right, as u(b) is unknown
		beta_n = 1
		if beta_order is None: beta_order = order
		pass
	else:
		beta_n = 0
		res[-1] = beta[1]

	for i in range(1,half_order):
		left_coeffs.append(fd.FDStencil(2, np.arange(-i, order+2-i), verbose=False))
		right_coeffs.append(fd.FDStencil(2, np.arange(-order-1+i, i+1), verbose=False))
	print(left_coeffs, right_coeffs)
	
	#Now we need to construct the banded matrix
	#The most lopsided an approximation could get is (a) a one-sided finite difference approximation of u' at an endpoint
	# or an approximation of u'' at an interior point immediately adjacent to the boundary. In both cases the number of points on one side
	# (and hence nonzero diagonals) is order, except for the very special case when order=2
	if order == 2: l = u = 1
	else: l = u = order

	dim = n - 1 + alpha_n + beta_n
	print(dim,l,u)
	#Store diagonals
	ab = np.zeros((l+u+1,dim))
	#Start with the center coefficients - the bulk of the system
	for k in range(len(center_coeffs)):
		#The affected diagonal is u - half_order + k
		diag_num = k - half_order
		start, stop = (0,dim) if not diag_num else ((-diag_num, dim) if diag_num < 0 else (0, -diag_num))
		ab[u+diag_num,start:stop] = center_coeffs[k]

	for i, coeffs in enumerate(left_coeffs):
		for k in range(len(coeffs)):
			diag_num = i-k+1 #If k=0 and i=0, it should be 1. If k increases, it decreases by 1. If i increases it increases by 1
			#If we have Dirichlet bc's, we should affect row i, column i + diag_num
			#If they are Nuemann, the row is i+1 as the first row has the approximation of u' at the boundary
			#So actually, in the case of Dirichelt BC's, the first coefficient always needs to be moved over to the right-hand side
			#because it is the coefficient by the constant term u(a)
			row = i + alpha_n
			j = row - diag_num
			if j < 0: continue
			ab[u+diag_num, j] = coeffs[k]

	for i, coeffs in enumerate(right_coeffs):
		for k in range(len(coeffs)):
			diag_num = -k-i+order #If i or k increases, the diagonal number goes down, and if both are 0 it should be order
			#If we have a Dirichlet about x=b, this will affect row dim-1-i
			#If they are Neumann, the affected row will be dim-2-i, as the last row will be for u'(b)
			row = dim - i - 1 - beta_n
			j = row - diag_num
			if j >= dim: continue
			ab[u+diag_num, j] = coeffs[k]

	dom = np.linspace(a,b,n+1)
	h = (b-a) / float(n)
	b = f(dom[1:-1]) * h**2
	if not alpha_n:
		#Figure out the coefficient by u(a) in the FD approximation for u''(a+h)
		i = 0
		while i < len(left_coeffs):
			b[i] -= alpha[1]*left_coeffs[i][0]
			i += 1
		b[i] -= center_coeffs[0] * alpha[1]
	if not beta_n:
		#Figure out the coefficient by u(b) in the FD approximation for u''(b-h)
		i = 0
		while i < len(right_coeffs):
			b[-1-i] -= beta[1]*right_coeffs[i][-1]
			i += 1
		b[-1-i] -= center_coeffs[-1] * beta[1]
	sol = solve_banded([l,u],ab,b)
	res[(1-alpha_n):dim+1-alpha_n] = sol
	return dom, res

import matplotlib.pyplot as plt

if __name__ == "__main__":
	f = lambda x: -np.sin(x)
	for order in range(2,8,2):
		dom, res = bvp(f, order=order, a=0, b=np.pi/2)
		plt.plot(dom, res, label="Order {}".format(order))
		print np.max(np.abs(np.sin(dom)-res)),order
	plt.legend()
	plt.show()

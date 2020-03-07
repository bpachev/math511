import numpy as np
import fdutil as fd
from scipy.linalg import solve_banded

def gen_grid(a,b,m,grid_type='uniform'):
	dom = np.linspace(0,1,m+1)
	if grid_type == 'uniform': return (b-a)*dom + a
	elif grid_type == 'rtlayer': return (a-b)*(dom-1)**2 + b
	elif grid_type == 'chebyshev': return -(b-a)/2. * np.cos(np.pi*dom) + (b+a)/2.
	else: raise ValueError("Unrecognized grid type {}".format(grid_type))

#Code for solving the boundary value problem epsu''-u = f(x) to second order, with a nonuniform grid
def prob1_bvp(f, grid, alpha=1, beta=3, eps=1e-2):
	#Assumes that grid[0] and grid[-1] are where the boundary conditions apply, and that the boundary conditions are Dirichlet
	n = len(grid) - 2 #number of unkowns
	ab = np.zeros((3,n)) #The matrix to store diagonals for scipy.solve_banded
	b = f(grid)[1:-1]
	for i in range(1,n+1):
		second_coeffs = fd.FDCoeffs(grid[i], grid[i-1:i+2],2)
		first_coeffs = fd.FDCoeffs(grid[i], grid[i-1:i+2],1)
#		print(second_coeffs, first_coeffs)
		coeffs = eps*second_coeffs - first_coeffs
		#main diagonal
		ab[1,i-1] = coeffs[1]
		#Upper diagonal
		if i < n: ab[0,i] = coeffs[2]
		else: b[-1] -= coeffs[2]*beta
		#Lower diagonal
		if i > 1: ab[2,i-2] = coeffs[0]
		else: b[0] -= coeffs[0]*alpha
	
	res = np.zeros(len(grid))
	res[0] = alpha
	res[-1] = beta
	res[1:-1] = solve_banded((1,1), ab, b)	
	return res

def l2_norm(dom, y):
	heights = (y[1:]**2+y[:-1]**2)/2.
	return np.sqrt(np.dot(heights, np.diff(dom)))

def make_table_data(u, f, sol_func,mvals):
	errs = []
	hvals = []
	orders = [0]
	for i,m in enumerate(mvals):
		grid, approx = sol_func(f,m)
		errs.append(l2_norm(grid, u(grid)-approx))
		hvals.append(1./m)
		if i: orders.append(np.polyfit(np.log(np.array(hvals)),np.log(np.array(errs)),1)[0])
	return [hvals, errs, orders]

from numpy.linalg import solve

def prob1_pseudospec(f,grid,alpha=1, beta=3, eps=1e-2):
	n = len(grid) - 2
	A = np.zeros((n,n))
	sol = f(grid)
	sol[0] = alpha
	sol[-1] = beta
	b = f(grid[1:-1])
	for i in range(1,n+1):
		second_coeffs = fd.stable_fdcoeffs(grid[i], grid, 2)
		first_coeffs = fd.stable_fdcoeffs(grid[i], grid, 1)
		coeffs = eps*second_coeffs - first_coeffs
		A[i-1] = coeffs[1:-1]
		b[i-1] -= alpha * coeffs[0] + beta * coeffs[-1]
	sol[1:-1] = solve(A,b)
	return sol

import matplotlib.pyplot as plt
from simple_bvp import bvp

if __name__ == "__main__":
	"""	f = lambda x: x-1 #Forcing function for problem 1
	eps = 1e-2
	u1 = lambda x: (1.-np.exp(x/eps))/(1-np.exp(1./eps)) * (3./2+eps) - 1./2*x**2 + (1-eps)*x + 1
	for t in ['uniform','rtlayer','chebyshev']:
#		g = gen_grid(0,1,40, t)
#		sol = prob1_pseudospec(f,g, eps=eps)
#		plt.plot(g, sol, label=t)
#		if t == 'chebyshev':
#			plt.plot(g, u1(g), label='true')
#		continue
		if t != 'chebyshev': continue
		def sol_func(f,m):
			g = gen_grid(0,1,m, t)
			sol = prob1_pseudospec(f,g, eps=eps)
			return g, sol
		print make_table_data(u1, f, sol_func, list(range(25,60,5)))
#	plt.legend()
#	plt.show()
	"""
	#Forcing func for problem 2
	f = lambda x: x
	true_sol = lambda x: x**3/6.+ x
	mvals = [40,60,20,160]
	for m in mvals:
		dom, sol = bvp(f, n=m, a=0,b=1, alpha=('N',1), beta=('N',3./2), order=2)
		plt.plot(dom, sol, label = "{} intervals".format(m))
	plt.plot(dom, true_sol(dom), label="Exact")
	plt.legend()
	plt.show()
	for m in mvals:
		dom, sol = bvp(f,n=m, a=0, b=1, alpha=('N',1), beta=('N',3./2), order=2, central=True)
		plt.plot(dom, sol, label = "{} intervals".format(m))
	plt.legend()
	plt.show()


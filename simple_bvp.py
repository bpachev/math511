import numpy as np
import fdutil as fd
#Would be interesting to compare solve_banded and spdiags + spsolve
#from scipy.sparse import spdiags
from scipy.linalg import solve_banded

def bvp(f, n = 20, a=0,b=1,order=2, alpha=('D', 0), beta=('D',1), alpha_order=None, beta_order=None, central=False):
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
	central - whether to use a centered difference for computing the Neumann boundary condition. Only valid for second-order
	Returns:
	x - the approximated solution
	"""
	#Ensure the order is even	
	order += order % 2
	half_order = int(order/2)
	center_coeffs = fd.FDStencil(2, np.arange(-half_order, half_order+1), verbose=False)
	left_coeffs = []
	right_coeffs = []
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
	#Here we generate the coefficients for the points close to the boundary where we can't use a centered formula
	for i in range(1,half_order):
		left_coeffs.append(fd.FDStencil(2, np.arange(-i, order+2-i), verbose=False))
		right_coeffs.append(fd.FDStencil(2, np.arange(-order-1+i, i+1), verbose=False))
	
	#Now we need to construct the banded matrix
	#The most lopsided an approximation could get is (a) a one-sided finite difference approximation of u' at an endpoint
	# or an approximation of u'' at an interior point immediately adjacent to the boundary. In both cases the number of points on one side
	# (and hence nonzero diagonals) is order, except for the very special case when order=2
	#l and u are number of lower and upper diagonals in the banded matrix
	if order == 2:
		l = u = 1
		if beta_order > 1: l = beta_order
		if alpha_order > 1: u = alpha_order
	else: l = u = order

	dim = n - 1 + alpha_n + beta_n
	#Store diagonals of the matrix A, so we can use Scipy to efficiently solve the system
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

	#Here we construct the right-hand-side of the system
	dom = np.linspace(a,b,n+1)
	h = (b-a) / float(n)
	b = np.zeros(dim)
	b[alpha_n:dim-beta_n] = f(dom[1:-1]) * h**2
	if not alpha_n:
		#Figure out the coefficient by u(a) in the FD approximation for u''(a+h)
		i = 0
		while i < len(left_coeffs):
			b[i] -= alpha[1]*left_coeffs[i][0]
			i += 1
		b[i] -= center_coeffs[0] * alpha[1]
	elif central:
		#we have (u_1 - u_-1)/2h = alpha[1] and (u_1-2u_0+u_-1) / h^2 = f(a)
		# 2*(u_1 - u_0) = h^2f(a) + 2h*alpha[1]
		b[0] = h**2 * f(dom[0]) + 2*h*alpha[1]
		ab[u,0] = 2
		ab[u-1,1] = 2
	else:
		coeffs = fd.FDStencil(1, np.arange(0, alpha_order+1), verbose=False)
		row = 0
		for k, c in enumerate(coeffs):
			col = k
			ab[u+row-col,col] = c
		b[0] = alpha[1]*h
	if not beta_n:
		#Figure out the coefficient by u(b) in the FD approximation for u''(b-h)
		i = 0
		while i < len(right_coeffs):
			b[-1-i] -= beta[1]*right_coeffs[i][-1]
			i += 1
		b[-1-i] -= center_coeffs[-1] * beta[1]
	elif central:
		#we have (u_n+1 - u_n-1)/2h = beta[1] and (u_n+1-2u_n+u_n-1) / h^2 = f(b)
		# 2*(u_n-1 - u_n) = h^2f(b) - 2h*beta[1]
		b[-1] = h**2 * f(dom[-1]) - 2*h*beta[1]
		ab[u,-1] = -2
		ab[u+1,-2] = 2
	else:
		coeffs = fd.FDStencil(1, np.arange(-beta_order, 1), verbose=False)
		b[-1] = beta[1]*h
		row = dim-1
		for k, c in enumerate(coeffs[::-1]):
			col = dim-1-k
			ab[u+row-col, col] = c
	#Here we use Scipy's specialized solver for banded systems - this is faster than a generic sparse solver
	#It does cause some indexing headaches, but is well worth it in terms of speed
	sol = solve_banded([l,u],ab,b)
	res[(1-alpha_n):dim+1-alpha_n] = sol
	return dom, res

def plot_table(cell_text, cols, title=""):
	fig, ax = plt.subplots()
	ax.axis('off')
	ax.axis('tight')
	ax.xaxis.set_visible(False) 
	ax.yaxis.set_visible(False)
	plt.title(title)
	plt.table(cellText = cell_text, colLabels=cols, loc='center')
	plt.show()


import matplotlib.pyplot as plt

def l2_norm(dom, y):
	heights = (y[1:]**2+y[:-1]**2)/2.
	return np.sqrt(np.dot(heights, np.diff(dom)))

def make_cell_text(vals):
	return [["{:.4e}".format(v[i]) for v in vals] for i in range(len(vals[0]))]

if __name__ == "__main__":
	ax, bx = 0, 3
	sigma, beta = -5, 3
	f = lambda x: np.exp(x)
	u = lambda x: np.exp(x) + (beta - np.exp(bx)) * (x-ax) + sigma - np.exp(ax)

	#Problem 2.4 (a)
	points = [[5,10,20,40],[10,20,40,80],[40,80,120,160]]
	for i,arr in enumerate(points):
		hvals = 1./np.array(arr)
		max1_errs = []
		max2_errs = []
		l2_errs1 = []
		l2_errs2 = []
		for n in arr:
			dom, res1 = bvp(f, n=n, a=ax, b=bx, alpha=('D',sigma), beta=('N', beta), order=2, central=True)
			dom, res2 = bvp(f, n=n, a=ax, b=bx, alpha=('D',sigma), beta=('N', beta), order=2)
			real = u(dom)
			max1_errs.append(np.max(np.abs(real-res1)))
			max2_errs.append(np.max(np.abs(real-res2)))
			l2_errs1.append(l2_norm(dom, real-res1))
			l2_errs2.append(l2_norm(dom, real-res2))

		cols = ["h", "central max norm","central l2","one-sided max norm", "one-sided l2"]
		vals = [hvals, max1_errs, l2_errs1, max2_errs, l2_errs2]
		text = make_cell_text(vals)
		plot_table(text, cols, "Experiment (a)({})".format(i+1))
		if i == len(points)-1:
			plt.title("L2-error plot")
			plt.loglog(hvals, l2_errs1, label = "central")
			plt.loglog(hvals, l2_errs2, label = "one-sided")
			plt.legend()
			plt.show()
	#2.4 (b)
	points = [[10,20,40,80],[40,80,100,120],[80,100,120,140,160,180,200]]
	for i,arr in enumerate(points):
		hvals = 1./np.array(arr)
		max1_errs = []
		max2_errs = []
		l2_errs1 = []
		l2_errs2 = []
		for n in arr:
			dom, res1 = bvp(f, n=n, a=ax, b=bx, alpha=('D',sigma), beta=('N', beta), order=6)
			dom, res2 = bvp(f, n=n, a=ax, b=bx, alpha=('D',sigma), beta=('N', beta), beta_order=4, order=6)
			real = u(dom)
			max1_errs.append(np.max(np.abs(real-res1)))
			max2_errs.append(np.max(np.abs(real-res2)))
			l2_errs1.append(l2_norm(dom, real-res1))
			l2_errs2.append(l2_norm(dom, real-res2))

		cols = ["h", "6-th order max norm","6-order l2","4-th order max norm", "4-th order l2"]
		vals = [hvals, max1_errs, l2_errs1, max2_errs, l2_errs2]
		text = make_cell_text(vals)
		plot_table(text, cols, "Experiment (b)({})".format(i+1))
		if i == len(points)-1:
			plt.title("L2-error plot")
			plt.loglog(hvals, l2_errs1, label = "6-th order")
			plt.loglog(hvals, l2_errs2, label = "4-th order")
			plt.legend()
			plt.show()

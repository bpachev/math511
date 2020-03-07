import numpy as np
from scipy.sparse import diags as sparse_diags
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg as sla

def make_5_point_diags(n, h):
	diags = {}
	diags[0] = -4. * np.ones(n*n)
	off_diag = np.ones(n*n-1)
	off_diag[n-1::n] = 0
	diags[-1] = diags[1] = off_diag
	diags[-n] = diags[n] = np.ones(n*(n-1))
	return diags

def make_9_point_diags(n, h):
	diags = {}
	diags[0] = -20. * np.ones(n*n)
	off_diag = 4*np.ones(n*n-1)
	off_diag[n-1::n] = 0
	diags[-1] = diags[1] = off_diag
	diags[-n] = diags[n] = 4*np.ones(n*(n-1))
	off_diag1 = np.ones(n*n-n+1)
	off_diag2 = np.ones(n*n-n-1)
	off_diag2[n-1::n] = 0
	off_diag1[0::n] = 0
	diags[n-1] = diags[1-n] = off_diag1
	diags[n+1] = diags[-1-n] = off_diag2
	return diags

def solve_hemholtz(f, boundary_func=lambda x,y: x*y, lap_f=lambda x: None, n=20,xint=(0,1), solver='spdiags', scheme='5-point', k=20):
	h = (xint[1]-xint[0])/float(n+1)

	if scheme == 'deferred':
		#Get the second-order approximation
		u_bar = np.zeros((n+2, n+2))
		u_bar[1:-1,1:-1] = solve_hemholtz(f, boundary_func, lap_f, n, xint,k=k).reshape((n,n))
		#Fill in the boundary data
		grid_1d = np.linspace(xint[0], xint[1], n+2)
		u_bar[0] = boundary_func(xint[0], grid_1d)
		u_bar[-1] = boundary_func(xint[1], grid_1d)
		u_bar[:,0] = boundary_func(grid_1d, xint[0])
		u_bar[:,-1] = boundary_func(grid_1d, xint[1])
		#Compute the xxyy derivative
		u_xx = (u_bar[:-2]+u_bar[2:]-2*u_bar[1:-1])/h**2
		u_xxyy = (u_xx[:,2:]+u_xx[:,:-2]-2*u_xx[:,1:-1])/h**2
	#First, construct the diagonals and the RHS
	#Here we assume that the domain is square
	b = np.zeros((n,n))
	yvals = np.linspace(xint[0], xint[1], n+2)[1:-1]
#	print h, yvals[1]-yvals[0]
	for i in xrange(n):
		xval = yvals[i]
		b[i] = f(xval, yvals)
		if scheme == '9-point':
			b[i] += h**2/12. * (lap_f(xval, yvals) - k**2*f(xval, yvals))
		elif scheme == 'deferred':
			b[i] += h**2/12. * (lap_f(xval, yvals) - k**2*f(xval, yvals) + k**4 * u_bar[i+1,1:-1] - 2*u_xxyy[i] )
	if scheme == '5-point' or scheme == 'deferred':
		b *= h**2
	else: b *= 6*h**2
#	print b
	if scheme=='5-point' or scheme == 'deferred':
		diags = make_5_point_diags(n, h)
		diags[0] += h**2 * k**2
		#Now we need to make use of the boundary conditions
		b[0] -= boundary_func(xint[0], yvals)
		b[-1] -= boundary_func(xint[1], yvals)
		b[:,0] -= boundary_func(yvals, xint[0])
		b[:,-1] -= boundary_func(yvals, xint[1])
	
	elif scheme=='9-point':
		diags = make_9_point_diags(n, h)
		diags[0] += 6 *((h*k)**2 - 1./12* (h*k)**4)
		#This is like the 5-point stencil
		b[0] -= 4*boundary_func(xint[0], yvals)
		b[-1] -= 4*boundary_func(xint[1], yvals)
		b[:,0] -= 4*boundary_func(yvals, xint[0])
		b[:,-1] -= 4*boundary_func(yvals, xint[1])
		#Here we handle the corners of the stencil
		vals = np.linspace(xint[0], xint[1], n+2)
		#Take care of effects to the top and bottom boundaries
		b[0] -= boundary_func(xint[0], vals[:-2]) + boundary_func(xint[0], vals[2:])
		b[-1] -= boundary_func(xint[1], vals[:-2]) + boundary_func(xint[1], vals[2:])
		#Handle the sides - be careful not to double count the corners
		b[1:,0] -= boundary_func(vals[1:-2], xint[0])
		b[:-1,0] -= boundary_func(vals[2:-1], xint[0])
		b[1:,-1] -= boundary_func(vals[1:-2], xint[1])
		b[:-1,-1] -= boundary_func(vals[2:-1], xint[1])

	else:
		raise ValueError("Unrecognized scheme {}".format(scheme))
	
	#Next, create the sparse matrix and solve the system
	print([(d, sum(diags[d]!=0)) for d in diags if isinstance(diags[d], np.ndarray)])
	offsets = sorted(diags.keys())
	mat = sparse_diags([diags[d] for d in offsets], offsets, format='csc')
#	print mat.todense()
	if solver == 'spdiags':
#		print b
#		print mat.shape
		return spsolve(mat, b.flatten())
	elif solver == 'solve_banded':
		ab, l_and_u = diags_to_banded(diags)
#		print b.flatten().shape, ab.shape
		return solve_banded(l_and_u,ab, b.flatten())
	else:
		return solver(mat, b.flatten())

def diags_to_banded(diags):
	offsets = sorted(diags.keys())
	l,u = abs(offsets[0]), offsets[-1]
	n = len(diags[0])
	ab = np.zeros((l+u+1,n))
	for d in diags:
		ab[u-d,max(0,d):min(n, n+d)] = diags[d]
	return ab, (l,u)

def l_inf_error(true, sol):
	return np.max(np.abs(true.flatten()-sol.flatten()))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_sol(sol, grid):
	X, Y = np.meshgrid(grid, grid)
	hf = plt.figure()
	ha = hf.add_subplot(111, projection='3d')
	ha.plot_surface(X,Y, sol)
	plt.show()

if __name__ == "__main__":
	from scipy.special import jve #Bessel function
	k = 20
	xpoints = 200
	f = lambda x,y: (k**2 -1) * (np.sin(y) + np.cos(x))
	lap_f = lambda x,y: -f(x,y) 
	true = lambda x,y: jve(0,k * (x**2+y**2)**.5) + np.sin(y) + np.cos(x)
	res = solve_hemholtz(f,boundary_func=true, n=xpoints-1, lap_f=lap_f, scheme='deferred', k= k)
	if len(res) == 2: res = res[0]
	grid = np.linspace(0,1,xpoints+1)
	
	true_sol = np.zeros((xpoints+1, xpoints+1))
	for i in xrange(xpoints+1):
		true_sol[i] = true(grid[i], grid)
	
	sol = np.zeros((xpoints+1, xpoints+1))
	sol[:,:] = true_sol
	sol[1:-1, 1:-1] = res.reshape((xpoints-1, xpoints-1))
	print "Max-norm error", l_inf_error(true_sol, sol)
	plot_sol(sol, grid)
	plot_sol(true_sol, grid)

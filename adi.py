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

def make_rhs(u,r,del_t):
	flat_u = u.flatten()
	b = np.full(u.size, del_t/2.) + (1-r) * flat_u
	n = u.shape[1]
	b[:-n] += (r/2.) * flat_u[n:]
	b[n:] += (r/2.) * flat_u[:-n]
	return b

def solve_poisson(nx, ny, del_t=.05, nits=40):
	"""
	Solves the equation u_t = u_xx + u_yy + 1 on the box [0,1]x[0,1].
	The initial and boundary data are all zero.
	"""
	final_res = np.zeros((nits, nx+2, ny+2))
	del_x = 1./(nx+1)
	del_y = 1./(ny+1)
	r_x = del_t/del_x**2
	r_y = del_t/del_y**2
	u = np.zeros(nx*ny)
	#Construct the diagonals for the tridiagonal matrices
	P_mid = np.full(nx*ny,1+r_x)
	P_high = np.full(P_mid.size-1, -r_x/2.)
	P_high[nx-1::nx] = 0
	#Symmetry
	P_low = P_high

	C_mid = np.full(nx*ny,1+r_y)
	C_high = np.full(C_mid.size-1, -r_y/2.)
	C_high[ny-1::ny] = 0
	#Symmetry
	C_low = C_high

	for n in xrange(nits):
		#Step 1, Predictor
		#The predictor step requires ordering first in y, then in x
		u_mat = u.reshape((nx,ny)).T
		b = make_rhs(u_mat, r_y, del_t)
		u = tridiag(P_low, P_mid, P_high, b)
		#Step 2, Corrector
		#Whereas the corrector step wants ordering first in x, then in y
		u_mat = u.reshape((ny, nx)).T
		d = make_rhs(u_mat, r_x, del_t)
		u = tridiag(C_low, C_mid, C_high, d)
		final_res[n,1:-1,1:-1] = u.reshape((nx,ny))

	return final_res

def exact_sol(x,y,t, terms=30):
	res = np.zeros((t.size, x.size, y.size))
	for k in xrange(terms):
		for l in xrange(terms):
			kc, lc = 1 + 2*k, 1 + 2*l
			exp_coeff = kc**2 + lc**2
			t_arr = 16./np.pi**4. * (1. - np.exp(-np.pi**2*exp_coeff*t)) / exp_coeff
			x_arr = np.sin(np.pi*kc*x)/kc
			y_arr = np.sin(np.pi*lc*y)/lc
			res += t_arr[:, None, None] * x_arr[None, :, None] * y_arr[None, None, :]
	return res

if __name__ == "__main__":
	nx = 25
	ny = 30
	del_t = .05
	nt = 40
	computed_res = solve_poisson(nx,ny,del_t=del_t, nits=nt)
	x = np.linspace(0,1,nx+2)
	y = np.linspace(0,1,ny+2)
	t = del_t * np.arange(1,nt+1)
	exact = exact_sol(x,y,t)
	X,Y = np.meshgrid(x,y)
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	hf = plt.figure()
	ha = hf.add_subplot(121, projection='3d')
	ha.plot_surface(X,Y, computed_res[1].T)
	ha = hf.add_subplot(122, projection='3d')
	ha.plot_surface(X,Y, exact[1].T)
	print np.max(np.abs(exact-computed_res))
	plt.show()

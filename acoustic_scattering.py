import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve
from fdutil import FDStencil
from scipy.special import jn, hankel1 #Bessel and hankel functions

def exact_soln(r, theta, terms = 30, r0=1, k = 2*np.pi):
	sln = np.zeros_like(theta).astype(np.complex128)
	for n in xrange(terms):
		eps = 2 if n else 1
		sln += 1j**n * jn(n,k*r0)/hankel1(n,k*r0)*hankel1(n,k*r)*np.cos(n*theta)
	return -sln

def solve_acoustic(r0=1,R=2,PPW=30,k=2*np.pi, sparse = False):
	bcs = np.zeros(PPW, dtype=np.complex128)
	thetas = np.linspace(0,2*np.pi,PPW+1)[:-1]
	bcs[:] = -np.exp(1j*r0*np.cos(thetas))
	#So first we need to determine the number of unkowns
	#This is going to be PPW * num distinct r
	#If we have PPW-1 interior radial points, then we end up with PPW^2 unknowns, provided we use only one term in the farfield expansion
	coeffs = FDStencil(1, np.array([-2,-1,0]), verbose=False).astype(np.complex128)
	#This gives us the coeffs for the equation a*U(R-2h, theta) + b*U(R-h,theta) - c * U(R, theta) = 0
	#which is what the ABC from the farfield expansion boils down to
	#Using more than one term will require us to treat the f_k explicitly, but in this case we don't have to
	coeffs[-1] -= 1j * k - 1./(2*R)
	n = PPW**2
	theta_h = 2*np.pi/PPW
	r_h = (R-r0)/float(PPW)
	if sparse:
		pass
	else:
		A = np.zeros((n,n), dtype=np.complex128)
		b = np.zeros(n,dtype=np.complex128)
		#Convert coordinate to offset in array
		conv = lambda i,j: i*PPW+j
		#Convert offset in array to coordinate
		inv = lambda k: divmod(k,PPW)
		#Add discrete polar Laplacian
		rs = np.linspace(r0,R,PPW+1)
		for i in xrange(PPW-1):
			for j in xrange(PPW):
				#Add in the theta derivative
				j_prev = (j-1+PPW) % PPW
				j_next = (j+1) % PPW
				pos = conv(i,j)
				A[pos,pos] += k**2 - 1./theta_h**2 - 1./r_h**2
				#Now handle the r derivatives
				A[pos,conv(i,j_prev)] += .5/theta_h**2
				A[pos,conv(i,j_next)] += .5/theta_h**2
				A[pos,conv(i+1, j)] += .5/r_h**2 + .5/r_h * 1./rs[i+1]
				c = .5/r_h**2 - .5/r_h * 1./rs[i+1]
				if i:
					A[pos, conv(i-1,j)] += c
				else:
					b[pos] -= c * bcs[j]
	
		#Now take care of the ABC (absorbing boundary condition)
		i = PPW-1
		for j in xrange(PPW):
			pos = conv(i,j)
			for k in xrange(len(coeffs)):
				A[pos, conv(i-k,j)] = coeffs[-1-k]
		
		true_soln = np.zeros(n+PPW, dtype=np.complex)
		true_soln[:PPW] = bcs
		true_soln[PPW:] = solve(A,b)
		return true_soln, rs, thetas

PPW = 45
R=5.
res,rs,thetas = solve_acoustic(PPW=PPW,R=R)

from matplotlib import pyplot as plt
plt.plot(thetas, np.abs(res[-PPW:]), label="Computed")
plt.plot(thetas, np.abs(exact_soln(R, thetas,terms=30)), label="Exact")
plt.legend()
plt.show()


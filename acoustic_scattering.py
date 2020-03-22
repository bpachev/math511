import numpy as np
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve
from fdutil import FDStencil
from scipy.special import jn, hankel1 #Bessel and hankel functions

def exact_soln(r, theta, terms = 30, r0=1, k = 2*np.pi):
	r = np.array(r)
	theta = np.array(theta)
	sln = np.zeros((len(r), len(theta)), dtype=np.complex128)
	for n in xrange(terms):
		eps = 2 if n else 1
		sln += eps* 1j**n * np.outer(jn(n,k*r0)/hankel1(n,k*r0)*hankel1(n,k*r), np.cos(n*theta))
	return -sln

def solve_acoustic(r0=1,R=2,PPW=30,k=2*np.pi, sparse = False):
	theta_h = 2*np.pi/(PPW)
	r_h = (R-r0)/float(PPW)
	bcs = np.zeros(PPW, dtype=np.complex128)
	thetas = np.linspace(0,2*np.pi,PPW+1)[:-1]
	bcs[:] = -np.exp(1j*k*r0*np.cos(thetas))
	#So first we need to determine the number of unkowns
	#This is going to be PPW * num distinct r
	#If we have PPW-1 interior radial points, then we end up with PPW^2 unknowns, provided we use only one term in the farfield expansion
	coeffs = FDStencil(1, np.array([-2,-1,0]), verbose=False).astype(np.complex128) / r_h
	#This gives us the coeffs for the equation a*U(R-2h, theta) + b*U(R-h,theta) - c * U(R, theta) = 0
	#which is what the ABC from the farfield expansion boils down to
	#Using more than one term will require us to treat the f_k explicitly, but in this case we don't have to
	coeffs[-1] -= 1j * k - 1./(2*R)
	n = PPW**2
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
			cur_r = rs[i+1]
			for j in xrange(PPW):
				#Add in the theta derivative
				j_prev = (j-1+PPW) % PPW
				j_next = (j+1) % PPW
				pos = conv(i,j)
				A[pos,pos] += k**2 - 2./theta_h**2/cur_r**2 - 2./r_h**2
				#Now handle the r derivatives
				A[pos,conv(i,j_prev)] += 1/theta_h**2/cur_r**2
				A[pos,conv(i,j_next)] += 1/theta_h**2/cur_r**2
				A[pos,conv(i+1, j)] += 1/r_h**2 + .5/r_h / cur_r
				c = 1/r_h**2 - .5/r_h / cur_r
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

def polar_to_cartesian(rs, thetas):
	return np.outer(rs, np.cos(thetas)), np.outer(rs, np.sin(thetas))


PPW = 40
R=2.
k = 2*np.pi
res,rs,thetas = solve_acoustic(PPW=PPW,R=R)

X,Y = polar_to_cartesian(rs,thetas)
#print X,Y
exact = exact_soln(rs, thetas, terms=30)
u_inc = np.exp(1j*k*X)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
hf = plt.figure()
ha = hf.add_subplot(121, projection='3d')
ha.plot_surface(X,Y, np.abs(u_inc+res.reshape(Y.shape)))
ha = hf.add_subplot(122, projection='3d')
ha.plot_surface(X,Y, np.abs(u_inc+exact))
plt.show()

plt.plot(thetas, np.abs(res[-PPW:]), label="Computed")
plt.plot(thetas, np.abs(exact[-1]), label="Exact")
plt.legend()
plt.show()


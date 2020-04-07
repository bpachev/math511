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

def solve_acoustic(r0=1,R=2,PPW=30,k=2*np.pi, scheme='farfield1', adjust_r = False):
	theta_h = 2*np.pi/(PPW)
	num_r = PPW if not adjust_r else int(3*R*PPW/k)
	r_h = (R-r0)/float(num_r)
	bcs = np.zeros(PPW, dtype=np.complex128)
	thetas = np.linspace(0,2*np.pi,PPW+1)[:-1]
	rs = np.linspace(r0,R,num_r+1)
	bcs[:] = -np.exp(1j*k*r0*np.cos(thetas))
	#Convert coordinate to offset in array
	conv = lambda i,j: i*PPW+j
	#Convert offset in array to coordinate
	inv = lambda k: divmod(k,PPW)
	coeffs = FDStencil(1, np.array([-2,-1,0]), verbose=False).astype(np.complex128) / r_h
	#This gives us the coeffs for the equation a*U(R-2h, theta) + b*U(R-h,theta) - c * U(R, theta) = 0
	#which is what the ABC from the farfield expansion boils down to
	def populate_interior():
		for i in xrange(num_r-1):
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
	if scheme == 'farfield1':
		#Using more than one term will require us to treat the f_k explicitly, but in this case we don't have to
		coeffs[-1] -= 1j * k - 1./(2*R)
		#So first we need to determine the number of unkowns
		#This is going to be PPW * num distinct r
		#If we have PPW-1 interior radial points, then we end up with PPW^2 unknowns, provided we use only one term in the farfield expansion
		n = PPW*num_r
		A = np.zeros((n,n), dtype=np.complex128)
		b = np.zeros(n,dtype=np.complex128)
		#Add discrete polar Laplacian
		populate_interior()
		#Now take care of the ABC (absorbing boundary condition)
		i = num_r-1
		for j in xrange(PPW):
			pos = conv(i,j)
			for k in xrange(len(coeffs)):
				A[pos, conv(i-k,j)] = coeffs[-1-k]
		
		true_soln = np.zeros(n+PPW, dtype=np.complex)
		true_soln[:PPW] = bcs
		true_soln[PPW:] = solve(A,b)
		return true_soln, rs, thetas
	elif scheme == 'farfield2':
		n = PPW*(num_r+1) #We handle the f's explicity
		A = np.zeros((n,n), dtype=np.complex128)
		b = np.zeros(n, dtype=np.complex128)
		populate_interior()
		c = np.exp(1j*k*R)/(k*R)**.5
		#The ABC conditions are a bit trickier
		#First add the continuity condition
		i = num_r-1
		for j in xrange(PPW):
			#Because of the ABC conditions, the last interior derivative will involve f_0 and f_1! A real Gotcha!
			#Fix them
			pos = conv(i,j)
			last_pos = conv(i-1,j)
			next_pos = conv(i+1,j)
			alpha = A[last_pos,pos]
			A[last_pos, pos] = alpha * c
			A[last_pos, next_pos] = alpha * c / (k*R) 
			#Coefficient by f_0(theta)
			A[pos,pos] = c * (coeffs[-1] -  (1j*k-1./(2*R)))
			#Coefficient by f_1(theta)
			A[pos, next_pos] = (c/(k*R)) * (coeffs[-1] - (1j*k-3./(2*R)))
			#Coefficients by Us
			for l in xrange(1,len(coeffs)):
				A[pos, conv(i-l,j)] = coeffs[-1-l]
		#Now take care of the relation 2i f_1 = f_0/4 + f_0''
		for j in xrange(PPW):
			pos = conv(num_r,j)
			A[pos,pos] = -2j
			A[pos, conv(num_r-1,j)] = 1./4 - 2./theta_h**2
			j_prev = (j-1+PPW) % PPW
			j_next = (j+1) % PPW
			A[pos, conv(num_r-1,j_prev)] = 1. / theta_h**2
			A[pos, conv(num_r-1,j_next)] = 1. / theta_h**2
		
		true_soln = np.zeros(PPW*(num_r+1), dtype = np.complex128)
		res = solve(A,b)
		true_soln[:PPW] = bcs
		true_soln[PPW:-PPW] = res[:-2*PPW]
		true_soln[-PPW:] = c*(res[-2*PPW:-PPW] + res[-PPW:]/(k*R))
		return true_soln, rs, thetas
			
	else: raise ValueError("Unrecognized Numerical Scheme {}".format(scheme))

def polar_to_cartesian(rs, thetas):
	return np.outer(rs, np.cos(thetas)), np.outer(rs, np.sin(thetas))

"""
PPW = 40
R=2.
k = 2*np.pi
res,rs,thetas = solve_acoustic(PPW=PPW,R=R, scheme='farfield2')

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
"""

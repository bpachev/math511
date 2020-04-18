import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

class gridGen():
	def __init__(self, x,y, branch_cut=None, omega=None, tol=1e-5, max_its=5000, multiply_connected=False):
		"""Set up a grid generation problem, where x and y contain the initial guess to the coordinates

		The shape of x and y must be the same, and the boundaries must be correct for the grid generation to work.
		If Omega is not specified, it is set to the optimal Omega for the Amsden-Hirst algorithm
		"""
		self.x = x
		self.y = y
		self.initial_x = x.copy()
		self.initial_y = y.copy()
		self.tol = tol
		self.omega = omega if omega is not None else self.AH_omega()
		self.max_its = max_its
		self.mc = multiply_connected
		if x.shape != y.shape:
			raise ValueError("x and y must have the same shape!")

	def AH_omega(self):
		n1,n2 = self.x.shape
		return 4./(2+(4-(np.cos(np.pi/n1)+np.cos(np.pi/n2))**2)**.5)

	def AH_step(self):
		old_x = self.x.copy()
		old_y = self.y.copy()
		n1,n2 = self.x.shape
		for i in range(1,n1-1):
			for j in range(1,n2-1):
				x_hat = (self.x[i-1,j]+self.x[i,j-1]+self.x[i+1,j]+self.x[i,j+1])/4.
				y_hat = (self.y[i-1,j]+self.y[i,j-1]+self.y[i+1,j]+self.y[i,j+1])/4.
				self.x[i,j] = self.omega * x_hat + (1-self.omega)*self.x[i,j]
				self.y[i,j] = self.omega * y_hat + (1-self.omega)*self.y[i,j]

		return self.check_convergence(old_x,old_y)

	def del_xi(self, u):
		return ((u[2:]-u[:-2])/2.)[:,1:-1]

	def del_eta(self, u):
		return ((u[:,2:]-u[:,:-2])/2.)[1:-1]

	def winslow_coeffs(self):
		x_xi = self.del_xi(self.x)
		x_eta = self.del_eta(self.x)
		y_xi = self.del_xi(self.y)
		y_eta = self.del_eta(self.y)
		return x_eta**2+y_eta**2, x_eta*x_xi + y_eta*y_xi, x_xi**2 + y_xi**2

	def Winslow_step(self):
		old_x = self.x.copy()
		old_y = self.y.copy()
		n1,n2 = self.x.shape

		alpha, beta, gamma = self.winslow_coeffs()

		#Handle the multiply connected case with a hole in the center
		if self.mc:
			x_xi = ((self.x[1] - self.x[-2])/2.)[1:-1]
			x_eta = (self.x[0,2:] - self.x[0,:-2])/2.
			y_xi = ((self.x[1] - self.y[-2])/2.)[1:-1]
			y_eta = (self.y[0,2:] - self.y[0,:-2])/2.
			a, b, g = x_eta**2+y_eta**2, x_eta*x_xi+y_eta*y_xi, x_xi**2 + y_xi**2
			for j in range(1,n2-1):
				x_hat = (a[j-1] * (self.x[-2,j]+self.x[1,j]) +g[j-1]*(self.x[0,j-1]+self.x[0,j+1]) -.5 * b[j-1] * (self.x[-2,j-1]-self.x[1,j-1]-self.x[-2,j+1]+self.x[1,j+1])) / (2 * (a[j-1]+g[j-1]))
				y_hat = (a[j-1] * (self.y[-2,j]+self.y[1,j]) +g[j-1]*(self.y[0,j-1]+self.y[0,j+1]) -.5 * b[j-1] * (self.y[-2,j-1]-self.y[1,j-1]-self.y[-2,j+1]+self.y[1,j+1])) / (2 * (a[j-1]+g[j-1]))
				self.x[0,j] = self.omega * x_hat + (1-self.omega)*self.x[0,j]
				self.y[0,j] = self.omega * y_hat + (1-self.omega)*self.y[0,j]

		for i in range(1,n1-1):
			for j in range(1,n2-1):
				a = alpha[i-1,j-1]
				b = beta[i-1, j-1]
				g = gamma[i-1, j-1]
				x_hat = (a * (self.x[i-1,j]+self.x[i+1,j]) +g*(self.x[i,j-1]+self.x[i,j+1]) -.5 * b * (self.x[i-1,j-1]-self.x[i+1,j-1]-self.x[i-1,j+1]+self.x[i+1,j+1])) / (2 * (a+g))
				y_hat = (a * (self.y[i-1,j]+self.y[i+1,j]) +g*(self.y[i,j-1]+self.y[i,j+1]) -.5 * b * (self.y[i-1,j-1]-self.y[i+1,j-1]-self.y[i-1,j+1]+self.y[i+1,j+1])) / (2 * (a+g))
				self.x[i,j] = self.omega * x_hat + (1-self.omega)*self.x[i,j]
				self.y[i,j] = self.omega * y_hat + (1-self.omega)*self.y[i,j]

		if self.mc:
			self.x[-1,:] = self.x[0]
			self.y[-1,:] = self.y[0]
		return self.check_convergence(old_x,old_y)
		

	def check_convergence(self, old_x, old_y):
		x_err = np.max(np.abs(self.x-old_x))
		y_err = np.max(np.abs(self.y-old_y))
#		print(x_err,y_err)
		return x_err < self.tol and y_err < self.tol

	def solve(self, algo='AH'):
		converged = False
		for k in range(self.max_its):
			if algo == 'AH':
				converged = self.AH_step()
			elif algo == 'Winslow':
				converged = self.Winslow_step()
			if converged:
				break

		if not converged:
			raise RuntimeWarning("No convergence after {} iterations".format(k))
		return k

	def reset(self):
		self.x = self.initial_x.copy()
		self.y = self.initial_y.copy()

	def test_omegas(self, omegas, algo="Winslow",plot=False):
		its = []
		for i, omega in enumerate(omegas):
			if i: self.reset()
			self.omega = omega
			its.append(self.solve(algo=algo))
		if plot:
			plt.plot(omegas, its)
			plt.xlabel("Omega")
			plt.ylabel("Iterations")
			plt.show()
		return its

	def jacobian(self, plot=False):
		x_xi = self.del_xi(self.x)
		x_eta = self.del_eta(self.x)
		y_xi = self.del_xi(self.y)
		y_eta = self.del_eta(self.y)
		jac = x_xi*y_eta - y_xi*x_eta
		if plot:
				hf = plt.figure()
				ha = hf.add_subplot(111, projection='3d')
				ha.plot_surface(self.x[1:-1,1:-1],self.y[1:-1,1:-1], jac)
				plt.show()

		inds = np.unravel_index(np.argmin(np.abs(jac)), jac.shape)
		min_jac = abs(jac[inds])
		x_coord = self.x[1:-1,1:-1][inds]
		y_coord = self.y[1:-1,1:-1][inds]
		return min_jac, x_coord, y_coord

	def plot(self, title=""):
		colors = np.zeros_like(self.x)+1.
		coords = np.zeros((2,self.x.size))
		coords[0] = self.x.flatten()
		coords[1] = self.y.flatten()
		mesh = plt.pcolormesh(self.x, self.y,colors, edgecolor='black', linewidth=1, cmap=ListedColormap(np.ones((256,4))))
		plt.title(title)
		plt.show()

def make_swan(n1,n2):
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	x_dom = np.linspace(0,1,n2)
	y_dom = np.linspace(0,1,n1)
	x[:,:] = x_dom
	for i in range(n2):
		y[:,i] = y_dom

	y[1:-1,1:-1]/=2
	#Now make sure the boundaries are correct
	y[-1] = 1. - 2*x_dom + 2*x_dom**2
	x[:,-1] = 1. + 2*y_dom - 2*y_dom**2
	return x,y


def make_dome(n1,n2):
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	x_dom = np.linspace(0,1,n2)
	y_dom = np.linspace(0,1,n1)
	x[:,:] = x_dom
	for i in range(n2):
		y[:,i] = y_dom

	#Now make sure the boundaries are correct
	y[-1] = 2. - 4*(x_dom-.5)**2
	return x,y

def make_rose(n1,n2):
	t = np.linspace(0,2*np.pi,n1)
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	r = .3*(2+np.cos(3*t))
	R = 2
	rs = r.reshape((n1,1)) + np.outer((R-r),np.linspace(0,1,n2))
	x = rs * np.cos(t).reshape((n1,1))
	y = rs * np.sin(t).reshape((n1,1))
	return x,y

def make_asteroid(n1,n2):
	t = np.linspace(0,2*np.pi,n1)
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	x[:,0] = .5*(3*np.cos(t)+np.cos(3*t))
	y[:,0] = .5*(3*np.sin(t)- np.sin(3*t))
	R = 6
	x[:,-1] = R*np.cos(t)
	y[:,-1] = R*np.sin(t)
	lambdas = np.linspace(0,1,n2)[1:-1]
	for i in range(n1):
		x[i,1:-1] = x[i,0] + lambdas*(x[i,-1]-x[i,0])
		y[i,1:-1] = y[i,0] + lambdas*(y[i,-1]-y[i,0])

	return x,y

simple_domains = {'swan':make_swan, 'dome':make_dome}
mc_domains = {'asteroid':make_asteroid, 'rose':make_rose}
import pandas as pd
def do_experiment(sizes, algo="Winslow", multiply_connected=False):
	domains = mc_domains if multiply_connected else simple_domains
	d_names = []
	d_sizes = []
	jac_vals = []
	jac_coords = []
	nits = []
	for d in domains:
		for s in sizes:
			d_names.append(d)
			d_sizes.append(s)
			gen = gridGen(*domains[d](*s), multiply_connected=multiply_connected)
			if algo == "Winslow": gen.omega=1.8
			nits.append(gen.solve(algo=algo))
			jac, x_coord, y_coord = gen.jacobian(plot=False)
			jac_vals.append(jac)
			jac_coords.append("({:.3f}, {:.3f})".format(x_coord, y_coord))
	return pd.DataFrame({"Domain":d_names, "Domain Size":d_sizes,"Min Jacobian":jac_vals, "Minimizer":jac_coords, "NumIts":nits})

if __name__ == "__main__":
	print(do_experiment([(71,21),(71,41)], algo="Winslow", multiply_connected=True))
#	print(4/(2+(4-(np.cos(np.pi/41.)+np.cos(np.pi/41.))**2)**.5))
#	gen = gridGen(*make_asteroid(41,21), omega=1.82, multiply_connected=True)
#	gen.plot()
#	print(gen.jacobian(plot=True))
#	print(gen.test_omegas([1.6,1.75,1.8,1.81,1.85,1.9], plot=True))
#	gen.plot()
#	gen = gridGen(*make_swan())


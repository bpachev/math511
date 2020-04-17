import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.colors import ListedColormap
#from mayavi import mlab

class gridGen():
	def __init__(self, x,y, branch_cut=None, omega=1., tol=1e-5, max_its=5000):
		"""Set up a grid generation problem, where x and y contain the initial guess to the coordinates

		The shape of x and y must be the same, and the boundaries must be correct for the grid generation to work.
		"""
		self.x = x
		self.y = y
		self.initial_x = x.copy()
		self.initial_y = y.copy()
		self.tol = tol
		self.omega = omega
		self.max_its = max_its
		if x.shape != y.shape:
			raise ValueError("x and y must have the same shape!")

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

	def check_convergence(self, old_x, old_y):
		return np.max(np.abs(self.x-old_x)) < self.tol and np.max(np.abs(self.y-old_y)) < self.tol

	def solve(self, algo='AH'):
		converged = False
		for k in range(self.max_its):
			if algo == 'AH':
				converged = self.AH_step()
			if converged:
				break

		if not converged:
			raise RuntimeWarning("No convergence after {} iterations".format(k))
		return k

	def plot(self):
		colors = np.zeros_like(self.x)+1.
		coords = np.zeros((2,self.x.size))
		coords[0] = self.x.flatten()
		coords[1] = self.y.flatten()
		mesh = plt.pcolormesh(self.x, self.y,colors, edgecolor='black', linewidth=1, cmap=ListedColormap(np.ones((256,4))))
		plt.show()

def make_swan(n1,n2):
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	x_dom = np.linspace(0,1,n1)
	y_dom = np.linspace(0,1,n2)
	x[:,:] = x_dom
	for i in range(n2):
		y[:,i] = y_dom

	#Now make sure the boundaries are correct
	y[-1] = 1. - 2*x_dom + 2*x_dom**2
	x[:,-1] = 1. + 2*y_dom - 2*y_dom**2
	return x,y


def make_dome(n1,n2):
	x = np.zeros((n1,n2))
	y = np.zeros((n1,n2))
	x_dom = np.linspace(0,1,n1)
	y_dom = np.linspace(0,1,n2)
	x[:,:] = x_dom
	for i in range(n2):
		y[:,i] = y_dom

	#Now make sure the boundaries are correct
	y[-1] = 2. - 4*(x_dom-.5)**2
	return x,y

if __name__ == "__main__":
	print(4/(2+(4-(np.cos(np.pi/41.)+np.cos(np.pi/41.))**2)**.5))
	gen = gridGen(*make_dome(41,41), omega=1.7)
#	gen.plot()
	print(gen.solve())
	gen.plot()


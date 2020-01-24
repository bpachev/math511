from fdutil import *
import numpy as np
import matplotlib.pyplot as plt

def prob1():
	print("Problem 1")
	FDStencil(1,np.arange(-2,1), verbose=True)
	print("The coefficient by the error is -1/3, as determined by hand.")

def prob2():
	print("\nProblem 2")
	inds = np.arange(-2,3)
	c = FDStencil(2,inds, verbose=True)
	print("We can see the coefficients are -1/12, 4/3, -5/2, 4/3 and -1/12")
	print("The coefficient of the error term is indeed 1/90, as computed by hand.")

	hvals = np.logspace(-1,-4, 13)
	#f(x) = sin(2x), so f(x)^(6) = -64*sin(2*x)
	x = 1.
	err_coeff = 64./90 * np.sin(2*x)
	x_vals = x+inds*hvals.reshape((len(hvals),1))
	u_vals = np.sin(2*x_vals)
	u_pp = u_vals.dot(c)/hvals**2
	real_u_pp = -4*np.sin(2*x)
	errors = u_pp-real_u_pp
	pred_errors = hvals**4 * err_coeff
	#Take every other one
	cell_text = []
	for i in range(0,len(hvals),2):
		order = 0 if not i else np.polyfit(np.log(hvals[:i]), np.log(np.abs(errors[:i])),1)[0]
		cell_text.append(["{:.4e}".format(hvals[i]), "{:.4e}".format(errors[i]), "{:.4e}".format(pred_errors[i]), "{:.4f}".format(order)])

	fig, ax = plt.subplots()
	ax.axis('off')
	ax.axis('tight')
	ax.xaxis.set_visible(False) 
	ax.yaxis.set_visible(False)
	plt.title("Table for 2(c). Note that I did part (d) :-)")
	plt.table(cellText = cell_text, colLabels=["h","error","predicted", "p"], loc='center')
	plt.show()
	plt.title("Plot of error for 2(c)")
	plt.loglog(hvals, np.abs(errors))
	plt.xlabel('h')
	plt.ylabel('error')
	plt.show()

def prob4():
	inds = np.arange(-4,1)
	print("Problem 4")
	print("Backwards difference approximation of u'''(x), with 5 points")
	FDStencil(3,inds, verbose=True)
	print("Book coefficients:", [3./2,-14./2,24./2,-18./2,5./2])
	print("\nCentered difference approximation of h'(x) with 4 points (but really 5)")
	FDStencil(1,np.array([-2,-1,0,1,2]))
	print("Book coefficients:", [1./12,-8./12,0,8./12,-1./12])

	print("\nBackwards difference approximation of u'(x) with 5 points")
	FDStencil(1,np.arange(-4,1))
	print("The centered difference formula is no better asymptotically, but has a slightly better prefactor on the error.")

	print("\nCentered finite difference formula for u''(x) with 7 points")
	FDStencil(2,np.arange(-3,4))


def FDnonuniform(f, fkp, fnp, fn1p, k, xb, xpts):
	c,err0, err1 = FDCoeffs(xb, xpts, k, ret_error=True)
	ukp = c.dot(f(xpts))
	print("u^({})({}) is approximately {}".format(k,xb,ukp))
	print("The true value is {}".format(fkp(xb)))
	print("The actual error is {}".format(ukp-fkp(xb)))
	print("The estimated error is {}\n".format(err0*fnp(xb)+err1*fn1p(xb)))

def prob5():
	#FDCoeffs is fdstencilnouniform
	print("Problem 5")
	c, err0, err1 = FDCoeffs(0,np.arange(-1,2),2, ret_error=True)
	print("Coeffs:", c)
	print("err0 {}, err1 {}".format(err0,err1))
	print("This is the usual centered difference formula for u''(x) with h=1. The first error is zero because the order of the derivative is even, improving the error by one order.")
	
	xb = .5
	k = 2
	f = lambda x: np.exp(x/3.)
	fkp = lambda x: np.exp(x/3.)/3.**k

	x_pts_list = [[xb-1e-1, xb, xb+1e-1],[xb-1e-1, xb-1e-3, xb, xb+1e-2], [-3,0,.5,1,4],[-3,0,.1/3,1,4]]
	for l in x_pts_list:
		n = len(l)
		x_pts = np.array(l)
		print("Using points", l)
		fnp = lambda x: np.exp(x/3.)/3.**n
		fn1p = lambda x: np.exp(x/3.)/3.**(n+1)
		FDnonuniform(f,fkp,fnp,fn1p, k, xb, x_pts)

	print("""It is unsurprising that more points with fixed step size leads to a better error. However, it is interesting to note that the simple centered difference formula with step size of 1e-1 is quite accurate, despite having only three points. This is probably explained by the advantage of a smaller step size as compared to the third and fourth variants. The second grid is the best, which is expected as it has step sizes on the order of 1e-3, and 1e-2, but is only one order better than the simple three-point centered difference.""")
prob1()
prob2()
prob4()
prob5()

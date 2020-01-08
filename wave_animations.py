import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Animation code taken from the following example
fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    return line,


def animate(i):
    line.set_ydata(np.sin(x + i / 100.))  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=.2, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()"""

def plot_wave(num_terms, L=30):
	x = np.linspace(0,L,100)
	n = np.arange(1,num_terms+1)
	coeffs = 9./(n**2*np.pi**2) * np.sin(n*np.pi/3)
	x_mat = np.sin(np.pi/L * n.reshape((num_terms,1)) * x) * coeffs.reshape((num_terms,1))

	f_init = np.zeros_like(x)
	mask = x<10
	f_init[mask] = x[mask]/10
	f_init[~mask] = 1 - x[~mask]/30
	fig, ax = plt.subplots()
	line, = ax.plot(x, f_init)
	ax.set_ylim(-1.3,1.3)

	def init():  # only required for blitting to give a clean slate.
		line.set_ydata([np.nan] * len(x))
		return line,
	
	def animate(i):
		t = i/10.
		line.set_ydata(np.dot(np.cos(2*n*np.pi*t/L),x_mat))  # update the data.
		return line,


	ani = animation.FuncAnimation(fig, animate, init_func=init, frames=600, interval=30, repeat=False, blit=True, save_count=50)
	plt.show()

if __name__ == "__main__":
	#Do animations
	plot_wave(30)
	pass

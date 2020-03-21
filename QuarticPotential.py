import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from matplotlib import rc

"""This is simple simulation code for a particle in a 1-D quantum mechanical 
x^4 potential."""

"""Relevant constants"""
N = 20
N_cor = 20
N_cf = 3000

a = 0.5
b = 0.01
c = 1.85
d = 1

eps = 1.5

average_x = np.zeros(N_cf)
average_V = np.zeros(N_cf)
action_list = np.zeros(N_cf)
X = []

def V(x):
	return d * (x**2 - c**2)**2 + b * x

"""Bounce solution solver. This gets fed into scipy.integrate.odeint"""
def bounce_EOM(z, t):
    #Here z is a vector such that z[0]=x' and z[1]=x.
    return np.asarray([4*d*z[1]**3 - 4*d*c**2*z[1] + b, z[0]])

def findIntersection(func1, func2, init_guess):
	return fsolve(lambda x: func1(x) - func2(x), init_guess)

def findBounce():
	x_0 = fmin(V, 1.7)
	min_val = V(x_0)
	boundary_negative = [0.0, -np.abs(fsolve(lambda x: V(x) - min_val, -1.8)[0])]
	_, bounce_sol_neg = odeint(bounce_EOM, boundary_negative, np.linspace(-4.0, 0, 1000)).T
	# boundary_positive = [0.0, np.abs(fsolve(lambda x: V(x) - min_val, -1.8)[0])]
	# _, bounce_sol_pos = odeint(bounce_EOM, boundary_positive, np.linspace(0, 3.0, 1000)).T
	return list(bounce_sol_neg)
	# return list(bounce_sol_pos) + list(bounce_sol_neg)

def plotBounce(bounce_sol):
	bounce_sol_plot, ax1 = plt.subplots()
	t = np.arange(0, N)
	ax1.plot(t, bounce_sol, 'b')

	# ax1.plot(euclidean_time, cool_closest_traj, 'r', label='closest trajectory after cooling')
	# ax1.set_xlabel("$t$")
	# ax1.set_ylabel("$x$")
	# ax1.legend(loc=1, prop={'size': 7})
	# bounce_sol_plot.savefig('bounce_solution_plot.png')

	plt.show()

def plotAction(action_list):
	action_plot, ax1 = plt.subplots()
	t = np.arange(0, N_cf)
	ax1.plot(t, action_list, 'b')
	plt.show()

def dS(j, x):
	jp = (j + 1)%N
	jm = (j - 1)%N
	return (-1.0)/2.0* x[j] * (x[jp] - 2.0*x[j] + x[jm])/a + (d*(x[j]**2 - c**2)**2 + b*x[j])*a

def action(x):
	S = 0.0
	for j in range(N):
		S += dS(j, x)
	return S

def update(x):
	for j in range (0, N):
		old_x = x[j]
		j_inc = (j + 1) % N
		j_dec = (j - 1) % N
		old_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		x[j] = x[j] + np.random.uniform(-eps, eps)
		new_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		delta_S = new_S - old_S
		if delta_S > 0 and np.exp(-delta_S) < np.random.uniform(0, 1):
			x[j] = old_x
	return x

def cool_update(x):
	x_cool = np.zeros(N_cf)
	for j in range (0, N):
		old_x = x[j]
		j_inc = (j + 1) % N
		j_dec = (j - 1) % N
		old_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		x[j] = x[j] + np.random.uniform(-eps, eps)
		new_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		delta_S = new_S - old_S
		#we accept updates that lower the action
		if delta_S < 0:
			x_cool[j] = x[j]
		else:
			x_cool[j] = old_x
	return x_cool

def metropolis(x, X, average_x, action_list):
	for j in range(0, 5 * N_cor):
		x = update(x)
	for alpha in range(1, N_cf):
		for j in range(N_cor):
			x = update(x)
			X.append(x)
			average_x[alpha] = np.average(x)
			action_list[alpha] = action(x) 
	return X, average_x, action_list

if __name__ == '__main__':
	x = np.full(N, fmin(V, 1.7))
	average_x[0] = np.average(x)
	action_list[0] = np.average(x)
	X.append(x)
	X, average_x, action_list = metropolis(x, X, average_x, action_list)
	#the peak action corresponds to a bounce
	bounce_traj = X[np.argmax(action_list)]
	bounce_traj_cooled = cool_update(bounce_traj)
import numpy as np
import matplotlib.pyplot as plt
from MonteCarloEstimator import *
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from matplotlib import rc

"""This is simple simulation code for a particle in a 1-D quantum mechanical 
x^4 potential."""

x_init = []
X = []
average_X = []
average_Y = []

"""Helper functions: initialization and bounce solution."""
def V(x):
	return d*(x**2 - c**2)**2 + b*x

def init_x(x, min_val):
	for i in range(N):
		val = np.random.uniform(low=min_val-1e-4, high=min_val+1e-4)
		x.append(val)
	return x

def bounce_EOM(z, t):
    """Here z is a vector such that z[0]=x' and z[1]=x."""
	return np.asarray([4*d*z[1]**3 - 4*d*c**2*z[1] + b, z[0]])

def findIntersection(func1, func2, init_guess):
	return fsolve(lambda x: func1(x) - func2(x), init_guess)

"""Gets the initital configuration"""
init_expect_x = fmin(V, 1.7)
x = init_x(x_init, init_expect_x[0])
X.append(list(x))
average_X.append(average(X[0]))

euclidean_time = range(N)

"""Thermalizes x values. Bascially, this discards the first several
randomly generated points (configurations) because they are atypical."""
for j in range(0, 5 * N_cor):
	update(x)

"""Looping over points, and getting new x values"""
for alpha in range(0, N_cf):
	for j in range(0, N_cor):
		update(x)
	X.append(list(x))
	average_X.append(average(x))

for i in average_X:
	average_Y.append(d*(i**2 - c**2)**2 + b*i)

"""Gets the bounce solution"""
min_func = V(init_expect_x)
result = fsolve(lambda x: V(x) - min_func, -1.8)

x0_neg = [0.0, -np.abs(result[0])]
euclidean_time_neg = np.linspace(-2.2, 0 , 1000)
_, bounce_sol_neg = odeint(bounce_EOM, x0_neg, euclidean_time_neg).T

x0_pos = [0.0, np.abs(result[0])]
euclidean_time_pos = np.linspace(0, 2.2 , 1000)
_, bounce_sol_pos = odeint(bounce_EOM, x0_pos, euclidean_time_pos).T

"""Finding the configuration corresponding to the maximum action"""
action_list = []
for n in range(N_cf):
	action = S(X[n])
	action_list.append((n, action))
sorted_action_list = sorted(action_list, key=lambda x: x[1], reverse=True)
max_idx = sorted_action_list[0][0]
max_config = X[max_idx]

"""This section is for plotting any computed quantities."""

"""NEEDS REFACTORING!"""

# Configuration corresponding to maximum action
# action_config, ax0 = plt.subplots()
# ax0.plot(euclidean_time, max_config)
# ax0.set_title('Plot of the maximizing configuration')
# ax0.set_xlabel('$t$')
# ax0.set_ylabel('$x[t]$')
# action_config.savefig('maximizing_configuration_plot.png')

# Average value of all configurations
# average_X_plot, ax1 = plt.subplots()
# ax1.plot(range(len(average_X)), average_X, 'b')
# average_X_plot.savefig('average_value_random_config.png')

# Plot of what each trajectory looks like
# trajectories, ax = plt.subplots()
# for i in range(N_cf):
# 	traj, = ax.plot(euclidean_time, X[i])
# 	plt.pause(1)
# 	traj.remove()

# Bounce solution plot
#bounce_sol_plot, ax1 = plt.subplots()
#bounce_sol = list(bounce_sol_pos) + list(bounce_sol_neg)
#interval = np.linspace(0, 19, 2000)
#ax1.plot(interval, bounce_sol, 'b', label='analytic bounce solution')

# Finding the bounce trajectory
# bounce_traj = []
# idx = 0
# while idx < 2000:
# 	bounce_traj.append(bounce_sol[idx])
# 	idx += 100
# corr_vals = []
# idx = 0
# for x in X:
# 	tup = (pearsonr(x, bounce_traj)[0], idx)
# 	corr_vals.append(tup)
# 	idx += 1
# corr_vals_sorted = sorted(corr_vals, key=lambda tup: tup[0])
# closest_traj_idx = corr_vals_sorted[len(corr_vals_sorted) - 1][1]
# closest_traj = X[closest_traj_idx]
# ax1.plot(euclidean_time, closest_traj, 'g', label='trajectory closest to bounce')

# #Running cooling sweeps
# cool_closest_traj = cool_update(closest_traj)
# for i in range(300):
# 	cool_closest_traj = cool_update(cool_closest_traj)
# ax1.plot(euclidean_time, cool_closest_traj, 'r', label='closest trajectory after cooling')
# ax1.set_xlabel("$t$")
# ax1.set_ylabel("$x$")
# ax1.legend(loc=1, prop={'size': 7})
# bounce_sol_plot.savefig('bounce_solution_plot.png')

# Tunneling process
# potential, ax = plt.subplots()
# x_plot = np.linspace(-4, 4, 1000)
# y_plot = d*(x_plot**2 - c**2)**2 + b*x_plot
# ax.plot(x_plot, y_plot)
# ax.set_title('Plot of the tunneling process: $V(x) = d(x^{2} - c^2)^{2} + b*x$')
# ax.set_xlabel('$x$')
# ax.set_ylabel('$V(x)$')
# x_val = 0
# y_val = 0
# for i in range(0, len(average_X)):
# 	x_val = average_X[i]
# 	y_val = average_Y[i]
# 	point, = ax.plot(x_val, y_val, 'go')
# 	plt.pause(0.01)
# 	point.remove()
# end_point, = ax.plot(x_val, y_val, 'go')

#plt.show()

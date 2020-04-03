from scipy.optimize import fmin
import numpy as np

"""This is simple simulation code for a particle in a 1-D quantum mechanical 
x^4 potential."""

"""Relevant constants"""
N = 30
N_cor = 20
N_cf = 4000
a = 1.0

b = 1./10.
c = 3.5
d = 1./120.

eps = 1.8
num_update = 0
num_fail = 0

average_x = np.zeros(N_cf)
average_V = np.zeros(N_cf)
action_list = np.zeros(N_cf)
X = np.empty(shape=(N_cf,N))

def V(x):
	return d * (x**2 - c**2)**2 + b * x

def dS(j, x):
	jp = (j + 1) % N
	jm = (j - 1) % N
	return (-1.0)/2.0* x[j] * (x[jp] - 2.0*x[j] + x[jm])/a + (d*(x[j]**2 - c**2)**2 + b*x[j])*a

def action(x):
	S = 0.0
	for j in range(N):
		S += dS(j, x)
	return S

def update(x):
	global num_update
	global num_fail
	for j in range (1, N - 1):
		old_x = x[j]
		j_inc = (j + 1)
		j_dec = (j - 1)
		old_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		x[j] = x[j] + np.random.uniform(-eps, eps)
		new_S = dS(j_dec, x) + dS(j, x) + dS(j_inc, x)
		delta_S = new_S - old_S
		num_update += 1
		if delta_S > 0 and np.exp(-delta_S) < np.random.uniform(0, 1):
			x[j] = old_x
			num_fail += 1
	return x

def metropolis(x, X, average_x, action_list, tv):
	global num_fail
	global num_update
	for j in range(0, 5 * N_cor):
		x = update(x)
	for alpha in range(1, N_cf):
		for j in range(N_cor):
			x = update(x)
			X[alpha] = x
			average_x[alpha] = np.average(x)
			action_list[alpha] = action(x)
			# if abs(np.average(x) - tv) < 0.5:
			# 	print("broken")
			# 	return X, average_x, action_list, num_update, num_fail
	return X, average_x, action_list

if __name__ == '__main__':
	x = np.full(N, fmin(V, 2))
	average_x[0] = np.average(x)
	action_list[0] = np.average(x)
	X[0] = x
	tv = fmin(V, -0.1)
	X, average_x, action_list = metropolis(x, X, average_x, action_list, tv)
	"""Saves data from this run externally so it can be accessed by another file"""
	np.savetxt("configuration_space.txt", X)
	np.savetxt("average_position.txt", average_x)
	np.savetxt("action_list.txt", action_list)
	print("Failed updates: ", (num_fail/num_update) * 100, "%")
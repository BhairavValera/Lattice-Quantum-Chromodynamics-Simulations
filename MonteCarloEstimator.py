import numpy as np
import matplotlib.pyplot as plt

"""This code computes correlation functions and is an implementation of the metropolis monte carlo algorithm. Adapted from Lattice QCD for Novices by Lepage."""

N = 20
N_cor = 20
N_cf = 1000

a = 0.5
b = 0.001
c = 1.85
d = 1.

eps = 1.5

x = np.zeros(N)
G = np.zeros((N_cf, N))
G_a = np.zeros((N_cf, N))

def compute_G(x, n):
	g = 0
	for j in range(0, N):
		g = g + x[j] * x[(j + n) % N]
	return g/N

def MCaverage(x,G):
	"""Initializes the x values"""
	for j in range(0, N):
		x[j] = 0.0

	"""Thermalizes x values. Bascially, this discards the first several
	randomly generated points (configurations) because they are atypical."""
	for j in range(0, 5 * N_cor): 
		update(x)

	"""Looping over points, and getting new x values for the compute_G method. Also,
	computing G for t+a for excitation energy."""
	for alpha in range(0, N_cf):
		for j in range(0, N_cor):
			update(x)
		for n in range(0, N):
			G[alpha][n] = compute_G(x, n)
		for n in range(0, N):
			t = n + 1
			G_a[alpha][n] = compute_G(x, t)
	return G, G_a

def S_a(j, x):
	jp = (j + 1)%N
	jm = (j - 1)%N
	return (-1.0)/2.0* x[j] * (x[jp] - 2.0*x[j] + x[jm])/a + (d*(x[j]**2 - c**2)**2 + b*x[j])*a

def S_a_cool(j, x):
	jp = (j + 1)%len(x)
	jm = (j - 1)%len(x)
	return (-1.0)/2.0* x[j] * (x[jp] - 2.0*x[j] + x[jm])/a + (d*(x[j]**2 - c**2)**2 + b*x[j])*a

def S(x):
	S = 0.0
	for j in range(N):
		S += S_a(j, x)
	return S

def update(x):
	for j in range (0, N):
		old_x = x[j]
		old_Sj = S_a(j, x)
		x[j] = x[j] + np.random.uniform(-eps, eps)
		dS = S_a(j, x) - old_Sj
		if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
			x[j] = old_x
	return x

def cool_update(x):
	x_cool = []
	for j in range(0, N):
		old_x = x[j]
		old_Sj = S_a_cool(j, x)
		x[j] = x[j] + np.random.uniform(-eps, eps)
		dS = S_a_cool(j, x) - old_Sj
		if dS < 0:
			x_cool.append(x[j])
		else:
			x_cool.append(old_x)
	return x_cool

def bootstrap(G, G_a):
	G_bootstrap = []
	G_bootstrap_a = []
	length = len(G)
	for i in range(0, length):
		alpha = int(np.random.uniform(0, length))
		alpha_a = int(np.random.uniform(0, length))
		G_bootstrap.append(G[alpha])
		G_bootstrap_a.append(G_a[alpha])
	return G_bootstrap, G_bootstrap_a

def bootstrap_run():
	g_bootstraps = []
	g_a_bootstraps = []
	for i in range(0, 100):
		G_bs, G_a_bs = bootstrap(G, G_a)
		g_bootstraps.append(average(G_bs))
		g_a_bootstraps.append(average(G_a_bs))
	g_bootstraps = np.asarray(g_bootstraps)
	g_a_bootstraps = np.asarray(g_a_bootstraps)
	return g_bootstraps, g_a_bootstraps

def bin(G, G_a, binsize):
	G_binned = []
	G_a_binned = []
	for i in range(0, len(G), binsize):
		avg_G_bin = 0
		for j in range(0, binsize):
			avg_G_bin = avg_G_bin + G[i + j]
		G_binned.append(avg_G_bin/binsize)
	for i in range(0, len(G_a), binsize):
		avg_G_a_bin = 0
		for j in range(0, binsize):
			avg_G_a_bin = avg_G_a_bin + G_a[i + j]
		G_a_binned.append(avg_G_a_bin/binsize)
	return G_binned, G_a_binned

"""Computes generic average of a N-d array. For example, 
it can compute the average of G over over each configuration."""
def average(X):
	x = sum(X)/len(X)
	return x

"""Computes the average of binned data"""
def bin_average(G_binned, G_a_binned):
	return 0

def deltaE(G, G_a):
	G = np.asarray(G)
	G_a = np.asarray(G_a)
	g = average(G)
	g_a = average(G_a)
	deltaE = np.log(g/g_a)/a
	return deltaE

def deltaE_bs(g, g_a):
	deltaE_bs = np.log(g/g_a)/a
	return deltaE_bs

def deltaE_bin(G_binned, G_a_binned):
	return 0

"""Computes generic standard deviation of a N-d array."""
def std_dev(X):
	sdev = np.absolute(average(X ** 2) - average(X)**2) ** 0.5
	return sdev

"""For the real values generated through Monte Carlo"""
# G, G_a = MCaverage(x, G)
# #x = np.arange(0, 3.5, a)
# E = deltaE(G, G_a)
# y = E[:7]

"""Bootstrap run"""
# std_bs = []
# g_bootstraps, g_a_bootstraps = bootstrap_run()
# deltaE_bs = deltaE_bs(g_bootstraps, g_a_bootstraps)
# deltaE_bs = deltaE_bs.T
# for i in deltaE_bs:
# 	std = np.std(i)
# 	std_bs.append(std)
# std_bs = std_bs[:7]

# plt.scatter(x, y_bin, edgecolors="white", s=50, c="royalblue")
# plt.errorbar(x, y_bin, yerr=std_bin, linestyle="None", ecolor="royalblue")
# plt.title("Extracted Energy vs t: error bars w/ Bootstrap process")
# Axes = plt.gca()
# Axes.set_ylim(-1, 3)
# plt.axhline(y=1, linestyle='--', color='green', xmin=0.0)
# plt.show()

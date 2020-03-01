"""This is based on the tutorial code found here: https://vegas.readthedocs.io/en/latest/tutorial.html#basic-integrals. Solution verified in Lattice QCD for Novices by Lepage"""

import vegas
import gvar
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline

boundary_list = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
results_list = []
x = []
x = np.asarray(x)

for i in boundary_list:
	x0 = i
	x8 = i
	def f(x):
		x = np.append(x, x0)
		x = np.insert(x, 0, x8)
		action_sum = 0
		for j in range(0, 8, 1):
			action_sum += (x[j + 1] - x[j])**2 + ((x[j])**2)/4
		return math.exp(-action_sum) * (1/(math.pi))**4
	integ = vegas.Integrator(7 * [[-5, 5]])
	result = integ(f, nitn=10, neval=10000)
	print(result.summary())
	print('result = %s    Q = %.2f' % (result, result.Q))
	integ_est = result.itn_results
	mean_results = []
	
	for i in integ_est:
		mean_results.append(gvar.mean(i))
	results_list.append(mean_results[9])

plt.plot(boundary_list, results_list)
plt.savefig('SHO.png')
plt.show()

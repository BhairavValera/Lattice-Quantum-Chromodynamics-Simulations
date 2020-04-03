import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from QuarticPotential import N, N_cf, dS, eps, a, b, c, d

def V(x):
    return d * (x**2 - c**2)**2 + b * x

"""Bounce solution solver. This gets fed into scipy.integrate.odeint"""
def bounce_EOM(z, t):
    #Here z is a vector such that z[0]=x' and z[1]=x.
    return np.asarray([4.0*d*z[1]**3 - 4.0*d*c**2*z[1] + b, z[0]])

def findAnalyticBounce():
    min_x = fmin(V, 5.0)
    min_V = V(min_x)
    #start at the right spot; same value of the false vacuum "up the hill"
    initial_position = fsolve(lambda x: V(x) - min_V, -1.8)[0]
    # initial_condition_neg = [0.0, np.abs(initial_position)]
    # _, bounce_sol_neg = odeint(bounce_EOM, initial_condition_neg, np.linspace(-4.0, 0, 1000)).T
    initial_condition_pos = [0.0, -np.abs(initial_position)]
    _, bounce_sol_pos = odeint(bounce_EOM, initial_condition_pos, np.linspace(0, N*a/2, 1000)).T
    # bounce_sol_neg = np.flip(bounce_sol_pos)
    # bounce_sol = list(bounce_sol_neg) + list(bounce_sol_pos)
    return bounce_sol_pos

def plotBounce(analytic_bounce_sol, X):
    """Preprocessing the analytic bounce solution for statistical comparisons"""
    bounce_sol_trunc = np.zeros(int(N/2))
    i = 0
    j = 0
    increment = int(1000/(N/2))
    traj = X[0][int(len(X[0])/2):]
    # print(analytic_bounce_sol[0])
    while j < 1000:
        if i == int(len(X[0])/2):
            break
        bounce_sol_trunc[i] = analytic_bounce_sol[j]
        j += increment
        i += 1
    """Finding the closest trajectory"""
    closest_trajectory = np.zeros(int(N/2))
    current_closeness = -np.infty
    for x in X:
        traj = x[int(len(X[0])/2):]
        new_closeness = pearsonr(bounce_sol_trunc, traj)
        if new_closeness[0] > current_closeness:
            closest_trajectory = traj

    func_plot, ax1 = plt.subplots()
    t = np.linspace(0.0, N*a/2, len(bounce_sol_trunc))
    ax1.set_xlabel("Euclidean Time")
    ax1.set_ylabel("x")
    ax1.plot(t, bounce_sol_trunc, 'b', label='Analytic Solution')
    ax1.plot(t, closest_trajectory, 'g', label='Monte Carlo Solution')
    ax1.legend(loc=1, prop={'size': 7})
    plt.grid(which='both')
    plt.show()
	# bounce_sol_plot.savefig('bounce_solution_plot.png')

def cool_update(x):
    x_cool = np.zeros(N_cf)
    for j in range (1, N - 1):
        old_x = x[j]
        j_inc = (j + 1)
        j_dec = (j - 1)
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

def plotArray(array):
    plot_, ax1 = plt.subplots()
    ax1.plot(range(len(array)), array)
    plt.show()

if __name__ == "__main__":
    X = np.loadtxt("configuration_space.txt")
    average_x = np.loadtxt("average_position.txt")
    action_list = np.loadtxt("action_list.txt")
    potential_bounces = X[500:1000]
    plotArray(average_x)
    plotBounce(findAnalyticBounce(), potential_bounces)
    
    # the peak action corresponds to a bounce
    # bounce_traj = X[np.argmax(action_list)]
    # bounce_traj_cooled = cool_update(bounce_traj)
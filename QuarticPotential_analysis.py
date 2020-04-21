import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import chisquare
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
    initial_condition_pos = [0.0, -np.abs(initial_position)]
    _, bounce_sol_pos = odeint(bounce_EOM, initial_condition_pos, np.linspace(0, N*a/2, 1000)).T
    bounce_sol_neg = np.flip(bounce_sol_pos)
    bounce_sol = list(bounce_sol_neg) + list(bounce_sol_pos)
    return bounce_sol

def bounce_fit(x, a, b, c, d):
    return a * (np.tanh(x//b) + c)**2 - d

def findMonteCarloBounce(analytic_bounce_sol, configuration_space, average_x, fv):
    """Preprocessing the analytic bounce solution for statistical comparisons"""
    bounce_sol_trunc = np.zeros(N)
    i = 0
    j = 0
    increment = int(len(analytic_bounce_sol)/N)
    while j < len(analytic_bounce_sol):
        if i == int(len(X[0])):
            break
        bounce_sol_trunc[i] = analytic_bounce_sol[j]
        j += increment
        i += 1

    """List keeps track of all potential bounces"""
    potential_bounce_idx = []
    for i in range(len(average_x)):
        if average_x[i] < -1:
            break
        elif np.abs(average_x[i] - fv) >= 0.4:
            if np.abs(min(configuration_space[i]) - min(bounce_sol_trunc)) <= 0.7:
                potential_bounce_idx.append(i)
            else:
                continue

    """Finding the closest trajectory"""
    closest_trajectory = np.zeros(N)
    potential_bounce_dict = {}
    for i in potential_bounce_idx:
        roll_value = np.argmin(configuration_space[i]) - np.argmin(bounce_sol_trunc)
        bounce_sol_roll = np.roll(bounce_sol_trunc, roll_value)
        chisquare_value = chisquare(bounce_sol_roll, configuration_space[i])[0]
        potential_bounce_dict[abs(chisquare_value)] = i
    smallest_chisquare = min(potential_bounce_dict.keys())
    closest_trajectory = configuration_space[potential_bounce_dict[smallest_chisquare]]
    return bounce_sol_trunc, closest_trajectory

def plotBounce(bounce_sol, closest_trajectory):
    func_plot, ax1 = plt.subplots()
    # np.roll(bounce_sol, 1)
    bounce_t = np.linspace(0, N*a/2, len(bounce_sol))
    monte_carlo_t = np.linspace(0, N*a/2, len(closest_trajectory))
    ax1.set_xlabel("Euclidean Time")
    ax1.set_ylabel("x")
    ax1.plot(bounce_t, bounce_sol, 'b', label='Analytic Solution')
    ax1.plot(monte_carlo_t, closest_trajectory, 'g', label='Monte Carlo Solution')
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
    ax1.set_xlabel("Monte Carlo Time")
    plt.show()

if __name__ == "__main__":
    X = np.loadtxt("configuration_space.txt")
    average_x = np.loadtxt("average_position.txt")
    action_list = np.loadtxt("action_list.txt")
    analytic_bounce = findAnalyticBounce()
    fv = fmin(V, 4)
    # plotArray(average_x)
    bounce_sol_trunc, closest_trajectory = findMonteCarloBounce(analytic_bounce, X, average_x, fv)
    plotBounce(analytic_bounce, closest_trajectory)
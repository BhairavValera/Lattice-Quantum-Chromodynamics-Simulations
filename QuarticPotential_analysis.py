import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import chisquare
from scipy.optimize import root_scalar
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
        chisquare_value = np.sum((bounce_sol_roll - configuration_space[i])**2)
        potential_bounce_dict[chisquare_value] = [i, roll_value]
    smallest_chisquare = min(potential_bounce_dict.keys())
    closest_trajectory = configuration_space[potential_bounce_dict[smallest_chisquare][0]]
    roll_value = potential_bounce_dict[smallest_chisquare][1]
    return bounce_sol_trunc, closest_trajectory, roll_value

def plotBounce(bounce_sol, closest_trajectory, roll_value):
    func_plot, ax1 = plt.subplots()
    bounce_t = np.linspace(0, N*a/2, len(bounce_sol))
    monte_carlo_t = np.linspace(0, N*a/2, len(closest_trajectory))
    closest_trajectory = np.roll(closest_trajectory, -roll_value)
    cooled_closest_trajectory = cool_update(closest_trajectory)
    ax1.set_xlabel("Euclidean Time")
    ax1.set_ylabel("x")
    ax1.plot(bounce_t, bounce_sol, 'b', label='Analytic Solution')
    ax1.plot(monte_carlo_t, closest_trajectory, 'g', label='Monte Carlo Solution')
    ax1.plot(monte_carlo_t, cooled_closest_trajectory, 'r', label='Cooled Monte Carlo Solution')
    ax1.legend(loc=1, prop={'size': 7})
    plt.grid(which='both')
    plt.show()
	# bounce_sol_plot.savefig('bounce_solution_plot.png')

def cool_update(x):
    x_cool = np.zeros(len(x))
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

def get_truncated_ensemble(average_x, configuration_space):
    average_x_truncated = []
    for i in range(0, len(average_x)):
        if np.abs(fv - average_x[i])[0] >= 3:
            average_x_truncated = average_x[0:i]
            break
    average_x_truncated = list(average_x_truncated)
    for i in range(len(average_x_truncated), len(average_x_truncated) - 1 + 100):
        average_x_truncated.append(average_x[i])
    ensemble_mean = np.mean(average_x_truncated)
    two_sigma = 2 * np.std(average_x_truncated)
    for i in range(len(average_x_truncated) - 1, 0, -1):
        if np.abs(average_x_truncated[i] - ensemble_mean) > two_sigma:
            average_x_truncated.pop(i)
        else:
            break
    return average_x_truncated

def calculate_decay_rate_trunc(average_x_truncated, configuration_space, fv, tv):
    root_result = root_scalar(V, bracket=[fv, tv])
    b = root_result.root
    truncated_ensemble_idx = range(len(average_x_truncated))
    count = 0
    for idx in truncated_ensemble_idx:
        for lattice_point in configuration_space[idx]:
            if lattice_point < b:
                count += 1
                break
    decay_rate = count/len(truncated_ensemble_idx)
    print(decay_rate)
    
def calculate_decay_rate_spatial(configuration_space):
    N_wall = 0
    root_result = root_scalar(V, bracket=[fv, tv])
    b = root_result.root
    for x in configuration_space:
        for lattice_point in x:
            if lattice_point < b:
                N_wall += 1
                break
    print(N_wall/N_cf)

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
    fv = fmin(V, 10)
    tv = fmin(V, -10)
    truncated_ensemble_idx = get_truncated_ensemble(average_x, X)
    calculate_decay_rate_trunc(truncated_ensemble_idx, X, fv, tv)
    calculate_decay_rate_spatial(X)
    # bounce_sol_trunc, closest_trajectory, roll_value = findMonteCarloBounce(analytic_bounce, X, average_x, fv)
    # plotBounce(analytic_bounce, closest_trajectory, roll_value)
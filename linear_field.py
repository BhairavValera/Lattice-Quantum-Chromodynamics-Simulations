import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import time
import os
import shutil
from MonteCarloEstimator import *
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from mpl_toolkits import mplot3d

"""Simple scalar field simulation code using metropolis algorithm for random generation of field configurations."""

start_time = time.time()

N = 60
N_cor = 20
N_cf = 1000
a = 1.0

c = 0.667558
d = 0.143112
epsilon = 0.0353678
msqcounter = -24.5 * d * c**2

S_bounce = 4
R = 6.0
w = 1.4

alpha = (np.pi * R)/(12 * S_bounce * w**3)
beta = (3 * S_bounce * w)/(2 * np.pi * R)
gamma = S_bounce/(2 * np.pi * R**2)
delta = np.sqrt((2 * np.pi * R)/(3 * S_bounce * w))

average_phi = np.zeros(N_cf)
action_list = np.zeros(N_cf)

"""Defining the potential in terms of local constants"""
def V_local_param(phi):
    return d * (phi**2 - c**2)**2 + epsilon * ((phi-c)/(2.0*c)) + msqcounter * phi**2


"""Defining the potential in terms of the bounce Radius, action, and bubble width"""
def V_physical(phi):
    return alpha * (phi**2 - beta)**2 + gamma * (phi * delta - 1)

def dS(phi, x, t):
    x_inc = (x + 1) % N
    x_dec = (x - 1) % N
    t_inc = (t + 1) % N
    t_dec = (t - 1) % N
    d2x = phi[x_inc][t] - 2.0 * phi[x][t] + phi[x_dec][t]
    d2t = phi[x][t_inc] - 2.0 * phi[x][t] + phi[x][t_dec]
    return (-1.0)/2.0 * phi[x][t] * (d2x + d2t) + V_local_param(phi[x][t]) * a**2

def action(phi):
    S = 0
    for x in range(N):
        for t in range(N):
            S += dS(phi, x, t)
    return S

def update_phi(phi):
    num_accepted = 0
    for x in range(0, N):
        for t in range(0, N):
            old_phi = phi[x][t]
            x_inc = (x + 1) % N
            x_dec = (x - 1) % N
            t_inc = (t + 1) % N
            t_dec = (t - 1) % N            
            old_S = dS(phi, x, t) + dS(phi, x_inc, t) + dS(phi, x_dec, t) + dS(phi, x, t_inc) + dS(phi, x, t_dec) 
            phi[x][t] = phi[x][t] + np.random.uniform(-eps, eps)
            new_S = dS(phi, x, t) + dS(phi, x_inc, t) + dS(phi, x_dec, t) + dS(phi, x, t_inc) + dS(phi, x, t_dec)
            delta_S = new_S - old_S
            if delta_S > 0 and np.exp(-delta_S) < np.random.uniform(0, 1):
                phi[x][t] = old_phi
                num_accepted += 1
    return phi, num_accepted

def metropolis(phi, average_phi, action_list):
    for j in range(0, 5 * N_cor):
        phi, num_accepted = update_phi(phi)
    for alpha in range(1, N_cf):
        for j in range(N_cor):
            phi, num_accepted = update_phi(phi)
        np.savetxt(path + "/config_" + str(alpha) + ".csv", phi, delimiter=' ')
        average_phi[alpha] = np.average(phi)
        action_list[alpha] = action(phi)
    return average_phi, action_list, num_accepted

"""Metropolis updating of the field, and saving the all of the configurations to .csv files.
We need a place to store configurations, so we make a directory to store it called
Confiurations/ to store them. If this directory already exists, then we delete it and
make a new one. Note that this will remove previous results. If your previous results
are useful to you, cache them somewhere so you don't lose them."""

if __name__ == "__main__":
    path = "./Configurations"
    access_rights = 0o755
    if (os.path.isdir(path) == True):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            print("Deletion of the directory %s failed. Check permissions." % path)
        else:
            print("Successfully deleted the directory %s " % path)
    try:
        os.mkdir(path, access_rights)
    except OSError:
        print("Creation of the directory %s failed. Directory may already exist" % path)
    else:
        print("Successfully created the directory %s " % path)

    phi_fv = fmin(lambda phi: V_local_param(phi), 1.1)[0]
    # print(phi_fv)
    phi = np.full((N, N), phi_fv)
    np.savetxt(path + "/config_" + str(0) + ".csv", phi, delimiter=' ')
    action_list[0] = action(phi)
    average_phi[0] = np.average(phi)
    average_phi, action_list, num_accepted = metropolis(phi, average_phi, action_list)
    np.savetxt("average_phi.csv", average_phi, delimiter=' ')
    np.savetxt("action_list.csv", action_list, delimiter=' ')
    print("Total time taken (sec): ", time.time() - start_time)
    print("Number of accepted metropolis steps: ", num_accepted)

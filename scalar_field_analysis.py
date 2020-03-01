import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.ndimage as ndimage
from findBounce import findBounce
from mpl_toolkits import mplot3d
from scipy.stats import pearsonr
from MonteCarloEstimator import cool_update
from linear_field import dS, N, eps, N_cf

"""Use this analysis code after scalar fields have been generated. To generate fields, run scalar_field.py, otherwise, the interpreter will throw errors."""

def cool_update_phi(phi):
    phi_cooled = np.zeros((N, N))
    for x in range(0, N):
        for t in range(0, N):
            old_phi = phi[x][t]
            old_S = dS(phi, x, t)
            phi[x][t] = phi[x][t] + np.random.uniform(-eps, eps)
            delta_S = dS(phi, x, t) - old_S
            """We accept Metropolis updates that lower the action"""
            if delta_S < 0:
                phi_cooled[x][t] = phi[x][t]
            else:
                phi_cooled[x][t] = old_phi
    return phi_cooled

def plotAction(action_list):
    t = np.arange(0, N_cf)
    action_plot, ax0 = plt.subplots()
    ax0 = plt.gca()
    ax0.plot(t, action_list)
    ax0.set_xlabel("Monte Carlo Time (t)")
    ax0.set_ylabel("Action (S)")
    ax0.axvline(x=149, c='g')
    action_plot.savefig('field_theory_action.png')
    plt.show()

def plotAveragePhi(average_phi):
    t = np.arange(0, len(average_phi))
    action_plot, ax1 = plt.subplots()
    ax1 = plt.gca()
    ax1.plot(t, average_phi)
    action_plot.savefig('average_phi.png')
    plt.show()

def plotBounce(bounceConfig):
    ax = plt.axes(projection='3d')
    x = np.arange(N)
    t = np.arange(N)
    X, T = np.meshgrid(x, t)
    ax.plot_surface(X, T, bounceConfig, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel("X")
    ax.set_ylabel("T")
    ax.set_zlabel("Phi")
    plt.show()

if __name__ == "__main__":
    path = "./Configurations"
    action_list = np.loadtxt(open("action_list.csv"), delimiter=' ')
    average_phi = np.loadtxt(open("average_phi.csv"), delimiter=' ')
    bounceConfig = np.loadtxt(open(path + "/config_" + str(149) + ".csv"), delimiter=' ')
    bounceConfigFlat = bounceConfig.flatten()
    bounceConfigFlat = np.roll(bounceConfigFlat, len(bounceConfigFlat) - np.argmin(bounceConfigFlat))
    bounceConfig = np.reshape(bounceConfigFlat, (N, N))
    bounceConfigCooled = cool_update_phi(bounceConfig)
    plotAction(action_list)
    plotBounce(bounceConfigCooled)

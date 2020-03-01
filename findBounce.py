import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from scipy.stats import linregress
from mpl_toolkits import mplot3d

"""Finding the bounce solution. Analytical solution adapted from Efficient numerical solution to
vacuum decay with many fields by Ali Masoumi et. al. Shooting method algorithm is still in development
as is the choice of the initial bounce profile."""

epsilon = 1.0
c = 1.5
d = 1.7

def V(phi):
    return (d * (phi**2 - c**2)**2 + epsilon * ((phi + c)/(2.0 * c)))

def phi_r_small(r, r1, A, B, phi_left):
    return phi_left + (A/B) * ((sp.special.iv(0, r * np.sqrt(B)))/(sp.special.iv(0, r1 * np.sqrt(B))) - 1)

def phi_r_small_prime(r, r1, A, B):
    return (A * sp.special.iv(1, r * np.sqrt(B)))/(np.sqrt(B) * sp.special.iv(0, r1 * np.sqrt(B)))

def phi_r_large(r, B, C, phi_fv):
    return phi_fv + C * sp.special.kv(0, r * np.sqrt(B))

def phi_r_large_prime(r, B, C):
    return -np.sqrt(B) * C * sp.special.kv(1, r * np.sqrt(B))

def calculate_C(phi_right, phi_fv, B, r_right):
    return (phi_right - phi_fv)/(sp.special.kv(0, np.sqrt(B) * r_right))

def bounce_EOM(z, r):
	"""Here z is a vector such that z[0]=phi' and z[1]=phi."""
	return np.asarray([(-1/r)*z[0] + 4*d*z[1]**3 - 4*d*c**2*z[1] + epsilon/(2**c), z[0]])

def findBounce():
    phi_fv = fmin(lambda phi: V(phi), 1.4)[0]

    shooting_domain = np.linspace(11.95, 12.6, 3)
    r1 = shooting_domain[0]
    r2 = shooting_domain[1]
    r3 = shooting_domain[2]

    r_small = np.linspace(0, r1, 1000)
    r_large = np.linspace(r3, 30, 1000)
    phi_left = -1.30
    phi_right = 1.362
    A = 4 * d * phi_left**3 - 4 * c**2 * d * phi_left + (epsilon)/(2 * c)
    B = 12 * d * phi_left**2 - 4 * c**2 * d

    """For calculating C, 'r_right' must be sufficiently large such that the solution is at the
    outer edge of the bounce. we assume that to be the point r = 13"""
    C = calculate_C(phi_right, phi_fv, B, 13)

    """ We stich together analytical solutions close to the center of the bounce (small rho) together
    with solutions far away from the center (large rho) and solutions as determined
    by odeint (intermediate distance)."""

    """Small r solution"""
    phi_r_small_sol = phi_r_small(r_small, r1, A, B, phi_left)

    """Intermediate r. This is where we use the shooting method. 
    We shoot until until we're below a certain tolerance.
    phi_init is array like, containing [phi_prime0, phi0]. Run solver once, then if the diff
    is too high, run it again."""
    delta = 0
    r_inter_1 = np.linspace(r1 + delta, r2, 1000)
    r_inter_2 = np.linspace(r3 - delta, r2, 1000)
    phi_init_left = [phi_r_small_prime(r1, r1, A, B), phi_r_small(r1, r1, A, B, phi_left)]
    _, phi_left_sol = odeint(bounce_EOM, phi_init_left, r_inter_1).T
    phi_init_right = [phi_r_large_prime(r3, B, C), phi_r_large(r3, B, C, phi_fv)]
    _, phi_right_sol = odeint(bounce_EOM, phi_init_right, r_inter_2).T
    
    """Calculate the current difference between the end points. If it's smaller than 0.001, then
    we're okay. But if it isn't, change it slightly (positive or negative) and calculate a new diff.
    If the new_diff is smaller than the current diff, then we changed in the right way. If not, then
    change in the other direction(?)"""
    # current_diff = np.abs(phi_left_sol[len(phi_left_sol) - 1] - phi_right_sol[len(phi_right_sol) - 1])
    # while (current_diff > 0.0001):
    #     r_inter_1 = np.linspace(r1 + delta, r2, 1000)
    #     r_inter_2 = np.linspace(r3 - delta, r2, 1000)
    #     phi_init_left = [phi_r_small_prime(r1, r1, A, B), phi_r_small(r1, r1, A, B, phi_left)]
    #     _, phi_left_sol = odeint(bounce_EOM, phi_init_left, r_inter_1).T

    #     phi_init_right = [phi_r_large_prime(r3, B, C), phi_r_large(r3, B, C, phi_fv)]
    #     _, phi_right_sol = odeint(bounce_EOM, phi_init_right, r_inter_2).T
    #     new_diff = np.abs(phi_left_sol[len(phi_left_sol) - 1] - phi_right_sol[len(phi_right_sol) - 1])
    #     if (new_diff <= current_diff):
    #         delta -= 0.0001
    #         current_diff = new_diff
    #     elif (new_diff > current_diff):
    #         delta += 0.0001

    """Large r solution"""
    phi_r_large_sol = phi_r_large(r_large, B, C, phi_fv)

    return [r_small, phi_r_small_sol, r_inter_1, phi_left_sol, r_inter_2, phi_right_sol, 
            r_large, phi_r_large_sol]

def plotBounce(bounceComponents):
    _, ax0 = plt.subplots()
    ax0.set_xlabel('rho')
    ax0.set_ylabel('phi')
    ax0.plot(bounceComponents[0], bounceComponents[1], 'g')
    ax0.plot(bounceComponents[2], bounceComponents[3], 'g')
    ax0.plot(bounceComponents[4], bounceComponents[5], 'g')
    ax0.plot(bounceComponents[6], bounceComponents[7], 'g')
    plt.show()

if __name__ == "__main__":
    bounceComponents = findBounce()
    plotBounce(bounceComponents)
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as spla
import xml.etree.ElementTree as ET

#-------------------------------------------------------------------------------
def sweep(I, hr, q, sigma_t, mu, boundary):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones 
        hr:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
    Outputs:
        psi:             value of angular flux in each zone
    """

    assert(np.abs(mu) > 1e-10)

    # intialize psi
    psi = np.zeros(I)
    
    # set the inverse of the cell width
    ihr =  1/hr
    
    # determine if sweepting with positive or negative mu
    # perform sweep
    if (mu > 0): 
        psi_minus = boundary
        for i in range(I):
            psi[i] =  (q[i]*0.5 + mu*psi_minus*ihr)/(sigma_t[i] + mu*ihr)
            psi_minus = psi[i]
    else:
        psi_plus = boundary
        for i in reversed(range(I)):
            psi[i] = (q[i]*0.5 - mu*psi_plus*ihr)/(sigma_t[i] - mu*ihr)
            psi_plus = psi[i]
    return psi
#-------------------------------------------------------------------------------
def source_iteration(I, hr, q, sigma_t, sigma_s, N, BCs, tolerance = 1.0e-8, maxits = 100, LOUD=False ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:               number of zones 
        hr:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        sigma_s:         array of scattering cross-sections
        N:               number of angles
        tolerance:       the relative convergence tolerance for the iterations
        maxits:          the maximum number of iterations
        LOUD:            boolean to print out iteration stats
    Outputs:
        r:               value of center of each zone
        phi:             value of scalar flux in each zone
    """

    # intialize phi and phi old
    phi = np.zeros(I)
    phi_old = phi.copy()

    # initialize convergence boolean to False
    converged = False

    # initialize angular moments and weights
    # these weights have been verified to sum to 2
    MU, W = np.polynomial.legendre.leggauss(N)

    # initialize the iteration number
    iteration = 1

    # allocate memory for the error
    error = []

    # start source iteration
    while not(converged):

        phi = np.zeros(I)
        
        # sweep over each direction
        for n in range(N):
            tmp_psi = sweep(I,hr,q + phi_old*sigma_s,sigma_t,MU[n],BCs[n])
            phi += tmp_psi*W[n]
        
        # check convergence
        change = np.linalg.norm(phi-phi_old)/np.linalg.norm(phi)
        error.append(change)
        converged = (change < tolerance) or (iteration > maxits)

        # print out Iteration number and convergence
        if (LOUD>0) or (converged and LOUD<0):
            print("Iteration",iteration,": Relative Change =",change)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
                
        # increment iteration number
        iteration += 1

        # reset phi_old for next iteration
        phi_old = phi.copy()

    # determine center values for r
    r = np.linspace(hr/2,I*hr-hr/2,I)
    return r, phi, error
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Inputs
#-------------------------------------------------------------------------------

root = ET.parse('input.xml').getroot()

# get common inputs
for common in root.findall('common'):
    # the ourter radius of the sphere
    rb = common.find('outer_radius').text

    

print(rb)


# the number of directions (even only)
N = 8

# the number of cells
I = 1000

# the source type
# 1 = constant isotropic distributed source
# 2 = right isotropic boundary flux
# 3 = right anisotropic boundary flux
# 4 = constant isotropic distributed source / right isotropic boundary flux
# 5 = constant isotropic distributed source / right anisotropic boundary flux
type = 1

# source normalization method
# 1 = point value
# 2 = integrated source value
normalization = 1

# source normalization values
# for point, define the zeroth Legendre moment of the source and the zeroth Legendre moment of the incident angular flux
# for integral, define the total source and the half-range current
normalization_values = (1, 1)

# sigma_a and sigma_t
sigma_a = np.ones(I)*1.0
sigma_t = np.ones(I)*1.0

# Pn scattering order, K
K = 0

# if K is > 0, anisotropic cross section coefficients
sigma_s = np.zeros((I, (K+1)))
if (K>0):
    sigma_s[:, 0] = 1.0
    sigma_s[:, 1] = 1.0
else:
    sigma_s[:, 0] = 1.0


# Acceleration (none/DSA)
use_DSA = False

# convergence tolerance
tolerance = 1.0E-6







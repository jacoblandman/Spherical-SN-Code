# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as spla
import xml.etree.ElementTree as ET

#-------------------------------------------------------------------------------
def sweep(I, hr, q, sigma_t, mu, BCs, N, gamma, alpha, beta, w, A, V):
    """Compute a transport sweep for a given
    Inputs:
        I:               number of zones 
        hr:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        boundary:        value of angular flux on the boundary
        n:               current angle discretization
        N:               number of angle discretization
    Outputs:
        psi:             value of angular flux in each zone
    """

    assert(all(abs(n) > 1e-10 for n in mu))

    # intialize psi at the quadrature points
    psi_edge = np.zeros(((I+1), (N)))
    psi_hat = np.zeros((I, (N)))
    psi = np.zeros((I, (N)))
    
    # initialze psi at the quadrature edges
    psi_edge_edge = np.zeros(((I+1), (N+1)))
    psi_hat_edge = np.zeros((I, (N+1)))
    psi_angle_edge = np.zeros((I, (N+1)))
    
    # determine starting direction flux
    psi_edge_edge[I, 0] = BCs[0]
    for i in reversed(range(I)):
        psi_edge_edge[i, 0] = (q[i] + psi_edge_edge[i+1,0]*(1 - 0.5*sigma_t[i]*hr))/(1 + 0.5*sigma_t[i]*hr)
        psi_hat_edge[i,0] = psi_edge_edge[i+1, 0]*gamma[i, 0] + psi_edge_edge[i, 0]*(1 - gamma[i, 0])
        psi_angle_edge[i,0] = psi_edge_edge[i+1, 0]*gamma[i, 1] + psi_edge_edge[i, 0]*(1 - gamma[i, 1])
    
    
    #print psi_edge_edge[:,0]
    #print psi_hat_edge[:,0]
    #print psi_angle_edge[:,0]
    # perform sweeps
    for n in range(N):
        # for negative angle mu's
        if (mu[n] < 0):
            psi_edge[I, n] = BCs[(n+1)]
            # sweep through all cells
            for i in reversed(range(0, I)):
                if (i > 0):
                    C1 = -mu[n]*A[i] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(1 - gamma[i,0])/beta[n] + sigma_t[i]*V[i]*(1 - gamma[i,1])
                    C2 = mu[n]*A[i+1] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(gamma[i,0])/beta[n] + sigma_t[i]*V[i]*(gamma[i,1])
                    C3 = 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(1 - beta[n])*psi_angle_edge[i, n]/beta[n] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n]*psi_hat_edge[i,n]
                    psi_edge[i,n] = (q[i]*V[i] - psi_edge[i+1,n]*C2 + C3)/C1
                    psi_hat[i,n] = psi_edge[i+1, 0]*gamma[i, 0] + psi_edge[i, 0]*(1 - gamma[i, 0])
                    psi[i,n] = psi_edge[i+1, 0]*gamma[i, 1] + psi_edge[i, 0]*(1 - gamma[i, 1])
                    psi_hat_edge[i,(n+1)] = (psi_hat[i,n] - (1 - beta[n])*psi_hat_edge[i,n])/beta[n]
                
                    # see if original equation is satisfied
                    tmp_psi = (psi_hat[i,n] - (1 - beta[n])*psi_hat_edge[i,n])/beta[n]
                    left_side = mu[n]*(A[i+1]*psi_edge[i+1,n] - A[i]*psi_edge[i,n]) + 0.5*(A[i+1] - A[i])/w[n]*(alpha[n+1]*psi_hat_edge[i,n+1] - alpha[n]*psi_hat_edge[i,n])
                    right_side = q[i]*V[i] - sigma_t[i]*V[i]*psi[i,n]
                    print "Left side = " + str(left_side)
                    print "Right side = " + str(right_side)
                    assert(left_side == right_side)
                    
                 # do something different for the first cell
                else:
                    gamma_tilde = (psi_edge[1,n]*2/3 + psi_edge_edge[0,0]*1/3)/(psi_edge[1,n]*3/4 + psi_edge_edge[0,0]*1/4)
                    C1 = sigma_t[i]*V[i] + A[i+1]/(2*w[n])*alpha[n+1]*gamma_tilde/beta[n]
                    C2 = alpha[n]*psi_hat_edge[i,n] - mu[n]*A[i+1]*psi_edge[i+1,n] + A[i+1]/(2*w[n])*alpha[n+1]*(1 - beta[n])/beta[n]*psi_hat_edge[i,n]
                    print C2
                    psi[i,n] = (q[i]*V[i] + C2)/C1
                    psi_edge[i,n] = (psi[i,n] - psi_edge[i+1,n]*gamma[i,1])/(1 - gamma[i,1])
                    psi_hat[i,n] = (psi_edge[i+1,n]*gamma[i,0] + (1 - gamma[i,0])*psi_edge[i,n])
            print psi_edge

            # now use a different discretization to solve for the flux in the first cell
    
        # perform sweeps for negative angle mu's
        else:
            psi_edge[0, n] = psi_edge_edge[0, 0]
            # sweep through all cells
            for i in range(1, (I+1)):
                #print i
                break



    return psi
#-------------------------------------------------------------------------------
def source_iteration(I, hr, q, sigma_t, sigma_s, N, K, tolerance = 1.0e-8, maxits = 100, LOUD=False ):
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

    # determine center and edge values for r
    r_center = np.linspace(hr/2,I*hr-hr/2,I)
    r_edge = np.linspace(0, rb, (I+1))
    
    # initialize psi_edge, psi_hat, psi
    psi = np.zeros((I, N))
    
    # intialize phi and phi old
    phi = np.zeros((I, (K+1)))
    phi_old = phi.copy()
    phi_old_old = phi.copy()
    
    # set area and volume
    A, V = set_area_and_volume(r_edge, I)

    # set gamma 1 and gamma 2 for each zone used to calculate psi_hat and psi
    gamma = set_gammas(r_edge, hr, I)

    # initialize angular moments and weights
    # these weights have been verified to sum to 2
    MU, W = np.polynomial.legendre.leggauss(N)
    MU_edge= np.zeros((N+1))
    MU_edge[0] = -1.0
    sum = 0.0
    for i in range(1, (N+1)):
        MU_edge[i] = MU_edge[i-1] + W[i-1]

    # set alpha and beta
    alpha, beta = set_alpha_beta(N, MU, MU_edge, W)
    
    # initialize convergence boolean to False
    converged = False

    #print (MU)
    # determine boundary conditions
    BCs = set_boundaries(MU, W, N)
    #print BCs

    # initialize the iteration number
    iteration = 1

    # allocate memory for the error
    error = []

    # start source iteration
    while not(converged):
        
        phi = np.zeros((I, (K+1)))
        tmp_phi = phi.copy()
        source = get_source(I, K, sigma_s, phi_old)
        # sweep over each direction
        psi = sweep(I,hr,q+source,sigma_t,MU,BCs, N, gamma, alpha, beta, W, A, V)
#        for n in range(N):
#            print (n)
#            source = get_source(I, K, sigma_s, phi_old)
#            tmp_psi = sweep(I,hr,q + source,sigma_t,MU[n],BCs[n], n, N)
#            phi = increment_phi(I, K, tmp_phi, tmp_psi, W[n], MU[n])
#            tmp_phi = phi.copy()

        # check convergence
        max_relative_change = np.max(np.abs((phi[:,0] - phi_old[:,0])/phi[:,0]))
        spectral_radius = np.sum(np.abs(phi[:,0] - phi_old[:,0]))/np.sum(np.abs(phi_old[:,0] - phi_old_old[:,0]))
        relative_error = max_relative_change/(1 - spectral_radius)
        converged = (relative_error < tolerance) or (iteration > maxits)
        converged = True

        # print out Iteration number and convergence
        if (LOUD>0) or (converged and LOUD<0):
            print("Iteration",iteration,": Relative Change =",change)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
                
        # increment iteration number
        iteration += 1

        # reset phi_old for next iteration
        phi_old_old = phi_old.copy()
        phi_old = phi.copy()
    
    # after convergence, calculate desired outputs
    #calculate_outputs()

    return r_center, phi, error
#-------------------------------------------------------------------------------
def set_area_and_volume(r, I):
    """Perform source iteration for single-group steady state problem
    Inputs:
        r:                  edge values for the radius
        I:                  number of zones
    Outputs:
        A:                  Area at each edge
        V:                  Volume of each zone
    """
    
    # areas are at edges and volumes are at centers
    
    # allocate memory
    A = np.zeros((I+1))
    V = np.zeros((I))

    for i in range(I):
        A[i] = 4*np.pi*r[i]*r[i]
        V[i] = 4*np.pi/3*(r[i+1]**3 - r[i]**3)

    A[(I)] = 4*np.pi*r[(I)]*r[(I)]

    return A, V

#-------------------------------------------------------------------------------
def set_gammas(r, hr, I):
    """Perform source iteration for single-group steady state problem
    Inputs:
        r:                  edge values for the radius
        hr:                 cell size
        I:                  number of zones
    Outputs:
        gamma:              normalization values for psi and psi_hat
    """
    # there should be I gamma values
    # allocate memory
    gamma = np.zeros((I,2))
    for i in range(I):
        gamma[i,0] = (3*r[i] + 2*hr)/(3*(2*r[i] + hr))
        gamma[i,1] = (6*r[i]**2 + 8*r[i]*hr + 3*hr**2)/(4*(3*r[i]**2 + 3*r[i]*hr + hr**2))

    return gamma

#-------------------------------------------------------------------------------
def set_alpha_beta(N, mu, mu_edge, w):
    """Perform source iteration for single-group steady state problem
    Inputs:
        mu:                 quadrature angles
        mu:                 edge values of mu
        w:                  quadrature weights
        N:                  number of angles
    Outputs:
        alpha:              angle coefficient
        beta:               weights for angle quadratures
    """
    # there will be N+1 betas and alphas because the first angle is -1, which is not a quadrature angle
    # allocate memory
    alpha = np.zeros((N+1))
    beta = np.zeros((N))

    alpha[0] = 0.0
    for n in range(0, N):
        alpha[n+1] = alpha[n] - 2*mu[n]*w[n]
        beta[n] = (mu[n] - mu_edge[n])/(mu_edge[n+1] - mu_edge[n])

    alpha[N] = 0
    return alpha, beta
#-------------------------------------------------------------------------------
def set_boundaries(mu, w, N):
    """Perform source iteration for single-group steady state problem
    Inputs:
        W:                  quadrature weight
        mu:                 quadrature angle
        N:                  number of angles
    Outputs:
        BCs:                Boundary Conditions
    """
    sum = 0.0

    for n in range (N):
        if (mu[n] < 0.0):
            # we want the incoming angular flux to be positive so use absolute value of mu
            sum += abs(mu[n])*w[n]

    current_incoming = 1.0
    psi_incoming = current_incoming/sum
    # N/2 because there are N/2 quadratures in the negative direction plus the starting directionsss
    BCs = np.ones((N/2) + 1)*psi_incoming
    return BCs

#-------------------------------------------------------------------------------
def get_source(I, K, sigma_s, phi_old ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:                  number of zones
        K:                  scattering order
        phi_old:            scalar flux from previous time step in each zone
        sigma_s:            array of scattering cross-sections
    Outputs:
        source:             scattering source for new time step
    """
    source = np.zeros((I))
    if (K == 0): source = sigma_s[:, 0]*phi_old[:,0]

    return source

#-------------------------------------------------------------------------------

def increment_phi(I, K, tmp_phi, tmp_psi, W, mu):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:                  number of zones
        K:                  scattering order
        tmp_phi:            scalar flux from previous time step in each zone
        tmp_psi:            temporary angular flux for angle mu
        W:                  quadrature weight
        mu:                 quadrature angle
    Outputs:
        source:             scattering source for new time step
    """
    phi = np.zeros((I,(K+1)))
    if (K == 0): phi[:,0] = tmp_phi[:,0] + tmp_psi*W
    elif (K == 1):
        phi[:,0] = tmp_phi[:,0] + tmp_psi*W
        phi[:,1] = tmp_phi[:,1] + tmp_psi*W*mu

    return phi

#-------------------------------------------------------------------------------

# Inputs
root = ET.parse('input.xml').getroot()

# get common inputs
common = root.find('common')

# the outer radius of the sphere
rb = float(common.find('outer_radius').text)

# the number of directions (even only)
N = int(common.find('n_angles').text)

# the number of cells
I = int(common.find('n_cells').text)

# Acceleration (none/DSA)
use_DSA = (common.find('use_DSA').text).lower() in ['1', 'y', 'yes', 'true', 't']

# convergence tolerance
tolerance = float(common.find('tolerance').text)

# max iterations
maxits = int(common.find('max_iterations').text)


# get source inputs
source = root.find('source')

# the source type
# 1 = constant isotropic distributed source
# 2 = right isotropic boundary flux
# 3 = right anisotropic boundary flux
# 4 = constant isotropic distributed source / right isotropic boundary flux
# 5 = constant isotropic distributed source / right anisotropic boundary flux
type = int(source.find('source_type').text)
assert (type <= 5)

# source normalization method
# 1 = point value
# 2 = integrated source value
normalization = int(source.find('normalization_method').text)
assert (normalization <= 2)

# source normalization values
# for point, define the zeroth Legendre moment of the source and the zeroth Legendre moment of the incident angular flux
# for integral, define the total source and the half-range current
val1 = float(source.find('normalization_value_1').text)
val2 = float(source.find('normalization_value_2').text)
normalization_values = (val1, val2)

# get cross section inputs
xs = root.find('cross_sections')
# sigma_a and sigma_t
sigma_a = np.ones(I)*float(xs.find('sigma_a').text)
sigma_t = np.ones(I)*float(xs.find('sigma_t').text)
assert(all(i >= j for j in sigma_a for i in sigma_t))

# Pn scattering order, K
K = int(xs.find('scattering_order').text)

# if K is > 0, anisotropic cross section coefficients
sigma_s = np.zeros((I, (K+1)))
if (K>0):
    sigma_s[:, 0] = sigma_t - sigma_a
    for k in range (1, (K+1)):
        key = "sigma_" + str(k)
        
        # check existance of all the required scattering coefficients
        test = (xs.find(key))
        if (test == None):
            print ("Error: Did not input enough scattering coefficients into xml file")
            exit(1)

        sigma_s[:, k] = float(xs.find(key).text)
else:
    sigma_s[:, 0] = sigma_t - sigma_a


hr = rb/I
q = np.zeros((I))
r, phi, error = source_iteration(I, hr, q, sigma_t, sigma_s, N, K, tolerance = 1.0e-8, maxits = 100, LOUD=False )

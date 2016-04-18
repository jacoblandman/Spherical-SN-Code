# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as spla
import xml.etree.ElementTree as ET
import scipy
from scipy import special

# legendre stuff
# b = special.eval_legendre(k,mu)

#-------------------------------------------------------------------------------
def sweep(I, hr, q, sigma_t, mu, BCs, N, gamma, alpha, beta, w, A, V):
    """ Compute a transport sweep in spherical coordinate system
    Inputs:
        I:               number of zones 
        hr:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        BCs:             value of angular flux on the boundary
        N:               number of angle discretization
        gamma:           normalization values for psi and psi_hat
        alpha:           angle coefficient
        beta:            weights for anglular discretization
        w:               quadrature weights
        A:               Area at each edge of the mesh
        V:               Volume of each zone
    Outputs:
        psi:             value of volume weighted angular flux in each zone for each quadrature
        psi_edge:        value of the angluar flux at the mesh edges for each quadrature
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
        psi_edge_edge[i, 0] = (q[i, 0]*hr + psi_edge_edge[i+1,0]*(1 - 0.5*sigma_t[i]*hr))/(1 + 0.5*sigma_t[i]*hr)
        psi_hat_edge[i,0] = psi_edge_edge[i+1, 0]*gamma[i, 0] + psi_edge_edge[i, 0]*(1 - gamma[i, 0])
        psi_angle_edge[i,0] = psi_edge_edge[i+1, 0]*gamma[i, 1] + psi_edge_edge[i, 0]*(1 - gamma[i, 1])
    
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
                    C3 = 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(1 - beta[n])*psi_hat_edge[i, n]/beta[n] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n]*psi_hat_edge[i,n]
                    psi_edge[i,n] = (q[i, (n+1)]*V[i] - psi_edge[i+1,n]*C2 + C3)/C1
                    psi_hat[i,n] = psi_edge[i+1, n]*gamma[i, 0] + psi_edge[i, n]*(1 - gamma[i, 0])
                    psi[i,n] = psi_edge[i+1, n]*gamma[i, 1] + psi_edge[i, n]*(1 - gamma[i, 1])
                    psi_hat_edge[i,(n+1)] = (psi_hat[i,n] - (1 - beta[n])*psi_hat_edge[i,n])/beta[n]
                
                    # see if original balance equation is satisfied
                    #left_side = mu[n]*(A[i+1]*psi_edge[i+1,n] - A[i]*psi_edge[i,n]) + 0.5*(A[i+1] - A[i])/w[n]*(alpha[n+1]*psi_hat_edge[i,n+1] - alpha[n]*psi_hat_edge[i,n])
                    #right_side = q[i]*V[i] - sigma_t[i]*V[i]*psi[i,n]
                    #print "left side = " + str(left_side)
                    #print "right side = " + str(right_side)
                    #assert(np.abs(left_side - right_side) < 1E-11*np.abs(left_side))
                    
                # do something different for the first cell
                else:
                    gamma_tilde = (psi_edge[i+1,n]*gamma[0,0] + psi_edge_edge[0,0]*(1 - gamma[0,0]))/(psi_edge[i+1,n]*gamma[0,1] + psi_edge_edge[0,0]*(1 - gamma[0,1]))
                    C1 = sigma_t[i]*V[i] + A[i+1]/(2*w[n])*alpha[n+1]*gamma_tilde/beta[n]
                    C2 = A[i+1]/(2*w[n])*alpha[n]*psi_hat_edge[i,n] - mu[n]*A[i+1]*psi_edge[i+1,n] + A[i+1]/(2*w[n])*alpha[n+1]*(1 - beta[n])/beta[n]*psi_hat_edge[i,n]
                    psi[i,n] = (q[i, (n+1)]*V[i] + C2)/C1
                    psi_hat[i,n] = gamma_tilde*psi[i,n]
                    
                    # set psi_edge to the starting direction flux
                    psi_edge[i,n] = psi_edge_edge[0,0]
                    psi_hat_edge[i,(n+1)] = (psi_hat[i,n] - (1 - beta[n])*psi_hat_edge[i,n])/beta[n]
                    tmp_psi_edge = (psi[i,n] - psi_edge[i+1,n]*gamma[i,1])/(1 - gamma[i,1])
                    tmp_psi_edge2 = (gamma_tilde*psi[i,n] - psi_edge[i+1,n]*gamma[i,0])/(1 - gamma[i,0])
                    
                    # see if original balance equation is satisfied
                    #left_side = mu[n]*A[i+1]*psi_edge[i+1,n] + 0.5*A[i+1]*(alpha[n+1]*psi_hat_edge[i,n+1] - alpha[n]*psi_hat_edge[i,n])/w[n]
                    #right_side = q[i]*V[i] - sigma_t[i]*V[i]*psi[i,n]
                    #assert(np.abs(left_side - right_side) < 1E-11*np.abs(left_side))
    
        # perform sweeps for positive angle mu's
        else:
            psi_edge[0, n] = psi_edge_edge[0, 0]
            # sweep through all cells
            for i in range(0, (I)):
                C1 = -mu[n]*A[i] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(1 - gamma[i,0])/beta[n] + sigma_t[i]*V[i]*(1 - gamma[i,1])
                C2 = mu[n]*A[i+1] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(gamma[i,0])/beta[n] + sigma_t[i]*V[i]*(gamma[i,1])
                C3 = 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n+1]*(1 - beta[n])*psi_hat_edge[i, n]/beta[n] + 1/(2*w[n])*(A[(i+1)] - A[i])*alpha[n]*psi_hat_edge[i,n]
                psi_edge[i+1,n] = (q[i, (n+1)]*V[i] - psi_edge[i,n]*C1 + C3)/C2
                psi_hat[i,n] = psi_edge[i+1, n]*gamma[i, 0] + psi_edge[i, n]*(1 - gamma[i, 0])
                psi[i,n] = psi_edge[i+1, n]*gamma[i, 1] + psi_edge[i, n]*(1 - gamma[i, 1])
                psi_hat_edge[i,(n+1)] = (psi_hat[i,n] - (1 - beta[n])*psi_hat_edge[i,n])/beta[n]

                # see if original balance equation is satisfied
                #left_side = mu[n]*(A[i+1]*psi_edge[i+1,n] - A[i]*psi_edge[i,n]) + 0.5*(A[i+1] - A[i])/w[n]*(alpha[n+1]*psi_hat_edge[i,n+1] - alpha[n]*psi_hat_edge[i,n])
                #right_side = q[i]*V[i] - sigma_t[i]*V[i]*psi[i,n]
                #assert(np.abs(left_side - right_side) < 1E-11*np.abs(left_side))

    np.set_printoptions(formatter={'float': lambda x: "{0:0.15f}".format(x)})
    return psi, psi_edge

#-------------------------------------------------------------------------------
def source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = 1.0e-8, maxits = 100, LOUD=False ):
    """Perform source iteration for single-group steady state problem
    Inputs:
        I:                          number of zones
        rb:                         outer boundary
        q:                          source array
        sigma_t:                    array of total cross-sections
        sigma_a:                    array of absorption cross-sections
        sigma_s:                    array of scattering cross-sections
        N:                          number of angles
        K:                          scattering order
        type:                       source type
        normalization:              point or integral normalization
        normalization_values:       source and boundary condition values
        tolerance:                  the relative convergence tolerance for the iterations
        maxits:                     the maximum number of iterations
        LOUD:                       boolean to print out iteration stats
    Outputs:
        r_center:                   value of center of each zone
        phi:                        value of scalar flux in each zone
        total_out_flow:             the total out flow of particles out of the sphere
    """

    # determine center and edge values for r
    hr = rb/I
    r_center = np.linspace(hr/2,I*hr-hr/2,I)
    r_edge = np.linspace(0, rb, (I+1))
    
    # initialize psi_edge, psi_hat, psi
    psi = np.zeros((I, N))
    
    # intialize phi and phi old
    phi = np.zeros((I, (K+1)))
    phi_old = np.zeros((I, (K+1)))
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
    BCs, q = set_sources(MU, W, N, type, normalization, normalization_values, I, V)
    #print BCs

    # initialize the iteration number
    iteration = 1

    # start source iteration
    while not(converged):
        
        phi = np.zeros((I, (K+1)))
        
        # calculate scattering source
        source = get_source(I, K, sigma_s, phi_old, q, MU, N)
        
        # sweep over each direction
        psi, psi_edge = sweep(I,hr,source ,sigma_t,MU,BCs, N, gamma, alpha, beta, W, A, V)

        # calculate phi
        phi = calculate_phi(psi, N, W, MU, K)
        
        # do dsa
        if (use_DSA):
            delta_phi, phi_edge, mu_average = perform_DSA(phi, phi_old, A, V, sigma_a, sigma_s, K, gamma[:,1], hr, I, MU, W, N)
            # update phi
            phi[:,0] += delta_phi
            for n in range (N):
                if (MU[n] > 0):
                    psi_edge[I,n] += 0.5*(phi_edge[I] + 3*phi_edge[I]*mu_average*MU[n])
            print_balance_table(psi, psi_edge, phi, N, I, K, MU, A, V, sigma_a, W, q)
                
        # check convergence
        max_relative_change = np.max(np.abs((phi[:,0] - phi_old[:,0])/phi[:,0]))
        if (iteration > 1):
            spectral_radius = np.sum(np.abs(phi[:,0] - phi_old[:,0]))/np.sum(np.abs(phi_old[:,0] - phi_old_old[:,0]))
        if (iteration == 1 or iteration == 2):
            spectral_radius = 0.0
        relative_error = float(max_relative_change/(1 - spectral_radius))
        converged = (relative_error < tolerance) or (iteration > maxits)
        
        # print out Iteration number and convergence
        if (LOUD>0) or (converged and LOUD<0):
            print("Iteration",iteration,": Relative error =",relative_error)
        if (iteration > maxits):
            print("Warning: Source Iteration did not converge")
                
        # increment iteration number
        iteration += 1
        
        # reset phi_old for next iteration
        phi_old_old = phi_old.copy()
        phi_old = phi.copy()
    
    # after convergence, calculate desired outputs
    total_out_flow = print_balance_table(psi, psi_edge, phi, N, I, K, MU, A, V, sigma_a, W, q)
    
    return r_center, phi, total_out_flow
#-------------------------------------------------------------------------------
def perform_DSA(phi, phi_old, A, V, sigma_a, sigma_s, K, gamma, hr, I, mu, w, N):
    """ Compute a transport sweep in spherical coordinate system
    Inputs:
        I:               number of zones 
        hr:              size of each zone
        q:               source array
        sigma_t:         array of total cross-sections
        mu:              direction to sweep
        BCs:             value of angular flux on the boundary
        N:               number of angle discretization
        gamma:           only the r^2 weighted values
        alpha:           angle coefficient
        beta:            weights for anglular discretization
        w:               quadrature weights
        A:               Area at each edge of the mesh
        V:               Volume of each zone
    Outputs:
        psi:             value of volume weighted angular flux in each zone for each quadrature
        psi_edge:        value of the angluar flux at the mesh edges for each quadrature
    """

    phi_edge = np.zeros((I+1))
    M = np.zeros((I+1,I+1))
    b = np.zeros((I+1))
    D = np.zeros((I))
    
    # determine diffusion coefficient
    for i in range (I):
        if (K > 0): D[i] = 1/(3*(sigma_t[i] - sigma_s[i,1]))
        else:
            D[i] = 1/(3*sigma_t[i])

    # determine mu average (used for right boundary condition)
    mu_average = 0.0
    for n in range (N):
        if (mu[n] > 0):
            mu_average += mu[n]*w[n]

    # set matrix coefficients for first and last edges
    M[0, 0] = D[0]*A[1]/hr + (1.0 - gamma[0])*sigma_a[0]*V[0]
    M[0, 1] = (sigma_a[0]*V[0]*gamma[0] - D[0]*A[1]/hr)
    M[(I), (I-1)] = sigma_a[I-1]*V[I-1]*(1 - gamma[I-1]) - (A[I-1] + A[I])*D[I-1]/hr
    M[(I), (I)] = sigma_a[I-1]*V[I-1]*(gamma[I-1]) + (A[I-1] + A[I])*D[I-1]/hr + 2*A[I]*mu_average
    b[0] = sigma_s[0]*V[0]*(phi[0] - phi_old[0])
    b[I] = sigma_s[I-1]*V[I-1]*(phi[I-1] - phi_old[I-1])

    # set matrix coefficients or interior edges
    for i in range (0, I-1):
        # diagonal
        M[i+1,i+1] = sigma_a[i]*V[i]*gamma[i] + (1 - gamma[i+1])*sigma_a[i+1]*V[i+1] + (A[i+1] + A[i+2])*D[i+1]/hr + (A[i] + A[i+1])*D[i]/hr
        # negative diagonal
        M[i+1, i] = sigma_a[i]*V[i]*(1 - gamma[i]) - (A[i] + A[i+1])*D[i]/hr
        # positive diagonal
        M[i+1, i+2] = sigma_a[i+1]*V[i+1]*(gamma[i+1]) - (A[i+1] + A[i+2])*D[i+1]/hr
        b[i+1] = sigma_s[i]*V[i]*(phi[i] - phi_old[i]) + sigma_s[i+1]*V[i+1]*(phi[i+1] - phi_old[i+1])

    # solve for the edge values
    phi_edge = np.linalg.solve(M,b)

    # collapse phi_edge
    phi_error = np.zeros((I))
    for i in range (I):
        phi_error[i] = gamma[i]*phi_edge[i+1] + (1 - gamma[i])*phi_edge[i]
    return phi_error, phi_edge, mu_average


#-------------------------------------------------------------------------------
def set_area_and_volume(r, I):
    """Set the area and volume for the problem
        Inputs:
        r:              edge values for the radius
        I:              number of zones
    Outputs:
        A:              Area at each edge
        V:              Volume of each zone
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
    """set the gamma values for the normalization of psi and psi_hat
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
    """set alpha and beta for the angular coefficients in the transport sweep
    Inputs:
        N:                  number of angles
        mu:                 quadrature angles
        mu_edge:            edge values of mu
        w:                  quadrature weights
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
def set_sources(mu, w, N, type, normalization, normalization_values, I, V):
    """set the boundaries  and q based off of the problem inputs
    Inputs:
        mu:                         quadrature angle
        W:                          quadrature weight
        N:                          number of angles
        type:                       source type
        normalization:              point or integral normalization
        normalization_values:       source and boundary condition values
        I:                          number of cells
        V:                          volume within each cell
    Outputs:
        BCs:                        Boundary Conditions for each quadrature of the angular flux
        q:                          external source
    """
    
    # BCs is size N/2 + 1 because there are N/2 quadratures in the negative direction plus the starting direction
    # we want the incoming angular flux to be positive so use absolute value of mu
    
    if (type == 1):
        # constant source, q, and vacuum boundary
        BCs = np.ones((N/2) + 1)*0.0
        if (normalization == 1):
            # the input file contains q/cm^3
            q = np.ones((I))*normalization_values[0]
        else:
            q = np.zeros((I))
            for i in range (I):
                q[i] = normalization_values[0]/V[i]

    elif (type == 2):
        # zero source, right isotropic boundary flux
        q = np.zeros((I))
        if (normalization == 1):
            # input contains the incident scalar flux
            BCs = np.ones((N/2) + 1)*normalization_values[1]/2
        else:
            sum = 0.0
            for n in range (N):
                if (mu[n] < 0.0):
                    sum += abs(mu[n])*w[n]
                        
            current_incoming = normalization_values[1]
            psi_incoming = current_incoming/sum
            BCs = np.ones((N/2) + 1)*psi_incoming

    elif (type == 3):
        # zero source, right anisotropic boundary flux
        q = np.zeros((I))
        if (normalization == 1):
            # input contains the incident scalar flux
            BCs = np.zeros((N/2) + 1)
            BCs[1] = normalization_values[1]/w[0]
            # starting direction flux
            if (N == 2): BCs[0] = BCs[1]
            else:
                BCs[0] = BCs[1]*(-1 - mu[1])/(mu[0] - mu[1])
        else:
            BCs = np.zeros((N/2) + 1)
            BCs[1] = normalization_values[1]/(mu[0]*w[0])
            # starting direction flux
            if (N == 2): BCs[0] = BCs[1]
            else:
                BCs[0] = BCs[1]*(-1 - mu[1])/(mu[0] - mu[1])

    elif (type == 4):
        # constant source, q, and right isotropic boundary flux
        if (normalization == 1):
            # input contains 1/cm^3 and incident scalar flux
            q = np.ones((I))*normalization_values[0]
            BCs = np.ones((N/2) + 1)*normalization_values[1]/2
        else:
            q = np.zeros((I))
            for i in range (I):
                q[i] = normalization_values[0]/V[i]
            
            sum = 0.0
            for n in range (N):
                if (mu[n] < 0.0):
                    sum += abs(mu[n])*w[n]
            
            current_incoming = normalization_values[1]
            psi_incoming = current_incoming/sum
            BCs = np.ones((N/2) + 1)*psi_incoming

    else:
        # constant source, q, and right anisotropic boundary flux
        if (normalization == 1 ):
            # input contains 1/cm^3 and incident scalar flux
            q = np.ones((I))*normalization_values[0]
            BCs = np.zeros((N/2) + 1)
            BCs[1] = normalization_values[1]/w[0]
            # starting direction flux
            if (N == 2): BCs[0] = BCs[1]
            else:
                BCs[0] = BCs[1]*(-1 - mu[1])/(mu[0] - mu[1])
        else:
            # input contains total q
            q = np.zeros((I))
            for i in range (I):
                q[i] = normalization_values[0]/V[i]

            BCs = np.zeros((N/2) + 1)
            BCs[1] = normalization_values[1]/(mu[0]*w[0])
            # starting direction flux
            if (N == 2): BCs[0] = BCs[1]
            else:
                BCs[0] = BCs[1]*(-1 - mu[1])/(mu[0] - mu[1])

    return BCs, q

#-------------------------------------------------------------------------------
def get_source(I, K, sigma_s, phi_old, q, mu, N ):
    """determine the new total source (scattering + q)
    Inputs:
        I:                  number of zones
        K:                  scattering order
        sigma_s:            array of scattering cross-sections
        phi_old:            scalar flux from previous time step in each zone
        q:                  array of source within each zone
        mu:                 quadrature angles
        N:                  number of quadrature angles
    Outputs:
        source:             scattering source for new time step + q
    """
    source = np.zeros((I, N+1))
    for k in range (K+1):
        source[:,0] += 0.5*(2*k + 1)*(sigma_s[:,k]*phi_old[:,k])*special.eval_legendre(k,-1.0)
    source[:,0] += 0.5*q
    for n in range (1, N+1):
        for k in range (K+1):
            source[:,n] += 0.5*(2*k + 1)*(sigma_s[:,k]*phi_old[:,k])*special.eval_legendre(k,mu[n - 1])
        source[:,n] += 0.5*q
    return source

#-------------------------------------------------------------------------------

def calculate_phi(psi, N, w, mu, K):
    """calculate the flux using the quadrature formula
    Inputs:
    
        psi:                value of the volume-weighted angular flux in each cell
        N:                  number of angular quadratures
        w:                  quadrature weight
        mu:                 quadrature angle
        K:                  scattering order
    Outputs:
        phi:                value of the volume-weighted scalar flux in each zone
    """
    phi = np.zeros((I,(K+1)))

    for k in range (K+1):
        for n in range (N):
            phi[:,k] += psi[:,n]*w[n]*special.eval_legendre(k,mu[n])

    return phi

#-------------------------------------------------------------------------------

def print_balance_table(psi, psi_edge, phi, N, I, K, mu, A, V, sigma_a, w, q):
    """ calculate and print the balance information
    Inputs:
        psi:                volume weighted angular flux
        psi_edge:           angular flux at each edge
        phi:                volume-weighted scalar flux in each zone
        N:                  number of angles
        I:                  number of zones
        K:                  scattering order
        mu:                 quadrature angles
        A:                  area at each edge of the mesh
        V:                  Volume of each cell of the mesh
        sigma_a             array of the absorption cross section
        w:                  quadrature weights
        q:                  external source in each zone
    Outputs:
        total_out_flow:     total number of particles flowing out of the surface
    """
    print " "
    print "------ Balance Table ------"
    total_in_flow = 0.0
    total_out_flow = 0.0
    total_absorption = 0.0
    total_source_rate = 0.0
    for n in range (N):
        if (mu[n] < 0.0):
            total_in_flow += psi_edge[I, n]*A[I]*np.abs(mu[n])*w[n]
        else:
            total_out_flow += psi_edge[I, n]*A[I]*np.abs(mu[n])*w[n]
    for i in range (I):
            total_absorption += phi[i,0]*V[i]*sigma_a[i]
            total_source_rate += q[i]*V[i]

    print "Total in flow = " + str(total_in_flow)
    print "Total out flow = " + str(total_out_flow)
    print "Total absorption = " + str(total_absorption)
    print "Total source rate = " + str(total_source_rate)

    balance = ( total_in_flow + total_source_rate - total_out_flow - total_absorption ) / ( total_in_flow + total_source_rate )
    print "Balance = " + str(balance)

    print " "


    # check conservation of the first cell
#    total_in_flow = 0.0
#    total_out_flow = 0.0
#    total_absorption = 0.0
#    total_source_rate = 0.0
#    for n in range (N):
#        if (mu[n] < 0.0):
#            total_in_flow += psi_edge[1, n]*A[1]*np.abs(mu[n])*w[n]
#        else:
#            total_out_flow += psi_edge[1, n]*A[1]*np.abs(mu[n])*w[n]
#
#    total_absorption = phi[0,0]*V[0]*sigma_a[0]
#    total_source_rate = q[0]*V[0]
#
#    print " "
#    print "-- Balance Table First Cell --"
#    print "Total in flow = " + str(total_in_flow)
#    print "Total out flow = " + str(total_out_flow)
#    print "Total absorption = " + str(total_absorption)
#    print "Total source rate = " + str(total_source_rate)
#
#    balance = ( total_in_flow + total_source_rate - total_out_flow - total_absorption ) / ( total_in_flow + total_source_rate )
#    print "Balance = " + str(balance)

    return total_out_flow



#-------------------------------------------------------------------------------

# Inputs
def get_inputs(file):
    """ Get the inputs for the problem
    Inputs:
        file:                       string containing the name of the xml input file being read
    Outputs:
        rb:                         outer boundary of the problem
         N:                         The number of angular quadratures
         I:                         The number of cells
         use_DSA:                   whether or not using diffusion synthetic accerleration
         tolerance:                 tolerance for source iteration
         maxits:                    max number of iterations for source iteration
         type:                      source type
         normalization:             point or integral normalization
         normalization_values:      normalization values
         sigma_a:                   array of absorption cross section
         sigma_t:                   array of total cross section
         sigma_s:                   array of scattering cross section
         K:                         scattering order
    """
    root = ET.parse(file).getroot()

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
    # 1 = constant isotropic distributed source (vacuum condition)
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
    # for point, define the zeroth Legendre moment of the source q/cm^3 and the zeroth Legendre moment of the incident angular flux
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

    print_inputs(rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K)
    return rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K

#-------------------------------------------------------------------------------

# Print Inputs
def print_inputs(rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K):
    """ Get the inputs for the problem
    Inputs:
        rb:                         outer boundary of the problem
         N:                         The number of angular quadratures
         I:                         The number of cells
         use_DSA:                   whether or not using diffusion synthetic accerleration
         tolerance:                 tolerance for source iteration
         maxits:                    max number of iterations for source iteration
         type:                      source type
         normalization:             point or integral normalization
         normalization_values:      normalization values
         sigma_a:                   array of absorption cross section
         sigma_t:                   array of total cross section
         sigma_s:                   array of scattering cross section
         K:                         scattering order
    """
    
    print "     .-')     .-')  .-. .-')"
    print "( OO ).  ( OO ).\  ( OO )"
    print "(_)---\_)(_)---\_);-----.\\"
    print "/    _ | /    _ | | .-.  |"
    print "\  :` `. \  :` `. | '-' /_)"
    print " '..`''.) '..`''.)| .-. `."
    print ".-._)   \.-._)   \| |  \  |"
    print "\       /\       /| '--'  /"
    print " `-----'  `-----' `------'"
    
    print ""

    print "---------------------------------------- Input Table ---------------------------------------------".center(85)
    print "Outer Boundary:".rjust(31) + "|".center(31) + str(rb).ljust(20)
    print "Number of Angles:".rjust(31) + "|".center(31) + str(N).ljust(20)
    print "Number of Cells:".rjust(31) + "|".center(31) + str(I).ljust(20)
    print "Using DSA:".rjust(31) + "|".center(31) + str(use_DSA).ljust(20)
    print "Tolerance:".rjust(31) + "|".center(31) + str(tolerance).ljust(20)
    print "Max Iterations:".rjust(31) + "|".center(31) + str(maxits).ljust(20)

    if (type == 1): s = "Constant isotropic distributed source (vacuum boundary)"
    elif (type == 2): s = "Right isotropic boundary flux"
    elif (type == 3): s = "Right anisotropic boundary flux"
    elif (type == 4): s = "Constant isotropic distributed source / right isotropic boundary flux"
    else: s = "Constant isotropic distributed source / right anisotropic boundary flux"

    print "Source Type:".rjust(31) + "|".center(31) + str(s).ljust(20)

    if (normalization == 1): s = "Point"
    else: s = "Integral"

    print "Normalization Type:".rjust(31) + "|".center(31) + str(s).ljust(20)
    print "Source Value:".rjust(31) + "|".center(31) + str(normalization_values[0]).ljust(20)
    print "Boundary Value:".rjust(31) + "|".center(31) + str(normalization_values[1]).ljust(20)
    print "Scattering Order:".rjust(31) + "|".center(31) + str(K).ljust(20)
    print "Absorption Cross Section:".rjust(31) + "|".center(31) + str(sigma_a[0]).ljust(20)
    print "Total Cross Section:".rjust(31) + "|".center(31) + str(sigma_t[0]).ljust(20)
    for k in range ((K+1)):
        s = "Scattering Cross Section k = " + str(k) + ":"
        print s.rjust(31) + "|".center(31) + str(sigma_s[0,k]).ljust(20)
    print ""




#-------------------------------------------------------------------------------

# Problems runs!

# Problem 17 part a

# 50 cells
#file = 'problem17_parta.xml'
#rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
#r, phi, total_out_flow_50 = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )

#plt.figure()
#plt.plot(r, phi)
#plt.xlabel("radius")
#plt.ylabel("flux")
#
## 100 cells
#file = 'problem17_parta2.xml'
#rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
#r, phi, total_out_flow_100 = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )
#
#plt.plot(r,phi)
#
#phi_parta = np.zeros((I))
#phi_parta = phi.copy()
#
## 200 cells
#file = 'problem17_parta3.xml'
#rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
#r, phi, total_out_flow_200 = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )
#
#plt.plot(r,phi)
#
## calculate xi
#xi = np.abs(total_out_flow_50 - total_out_flow_100)/np.abs(total_out_flow_100 - total_out_flow_200)
#print "Xi = " +str(xi)
#
#
## Problem 17 part b
#file = 'problem17_partb.xml'
#rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
#r, phi, total_out_flow = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )
#
#plt.figure()
#plt.plot(r, phi)
#plt.xlabel("radius")
#plt.ylabel("flux")
#plt.ylim(0, 2)
#q = np.ones((I))
#plt.plot(r, q/sigma_a)
#
#
# Problem 17 part c
#file = 'problem17_partc.xml'
#rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
#r, phi, total_out_flow = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )
#
#plt.figure()
#plt.plot(r, phi_parta, label="Part a")
#plt.plot(r, phi[:,0], label="Higher order scatter")
#plt.xlabel("radius")
#plt.ylabel("flux")
#plt.legend(loc="best")
#
## Problem 17 part d
file = 'problem17_partd.xml'
rb, N, I, use_DSA, tolerance, maxits, type, normalization, normalization_values, sigma_a, sigma_t, sigma_s, K = get_inputs(file)
r, phi, total_out_flow = source_iteration(I, rb, sigma_t, sigma_a, sigma_s, N, K, type, normalization, normalization_values, tolerance = tolerance, maxits = maxits, LOUD=True )
#
#plt.figure()
#plt.plot(r, phi[:,0], label="")
#plt.xlabel("radius")
#plt.ylabel("flux")
#plt.legend(loc="best")
#plt.show()
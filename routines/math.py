import numpy as np 
from numba import njit
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.chebyshev import Chebyshev,cheb2poly

# ------------------------------------------------------------------------------------------------------------
# Exact matrix determinants and inverses for 3D matrices. X must be an np.ndarray of appropriate dimensions.
# ------------------------------------------------------------------------------------------------------------
def det2(X):
    return X[0,0]*X[1,1] - X[0,1]*X[1,0]

def det3(X):
    m0 = X[1,1]*X[2,2]-X[2,1]*X[1,2]
    m1 = X[1,0]*X[2,2]-X[2,0]*X[1,2]
    m2 = X[1,0]*X[2,1]-X[2,0]*X[1,1]
    return X[0,0]*m0 - X[0,1]*m1 + X[0,2]*m2

def inv3(X):
    det = det3(X) 
    cof = np.zeros((3,3))
    cof[0,0] = det2(X[np.ix_([1,2],[1,2])])
    cof[0,1] = - det2(X[np.ix_([1,2],[0,2])])
    cof[0,2] = det2(X[np.ix_([1,2],[0,1])])
    cof[1,0] = - det2(X[np.ix_([0,2],[1,2])])
    cof[1,1] = det2(X[np.ix_([0,2],[0,2])])
    cof[1,2] = - det2(X[np.ix_([0,2],[0,1])])
    cof[2,0] = det2(X[np.ix_([0,1],[1,2])])
    cof[2,1] = - det2(X[np.ix_([0,1],[0,2])])
    cof[2,2] = det2(X[np.ix_([0,1],[0,1])])
    return cof.T/det

# ------------------------------------------------------------------------------------------------------------
# Simple Integrals of Polynomial Functions of LogNormally Distributed Variables with mean mu, variance sigma2
# ------------------------------------------------------------------------------------------------------------
def LogNormalMoment(a,mu=0,sigma=1,Nquad=11):
    '''
    Compute E(Z**a) where logZ is normal by Gauss-Hermite Quadrature
    '''
    nodes,weights = hermgauss(Nquad)
    nodes = np.sqrt(2)*sigma*nodes + mu
    f = np.exp(nodes*a)
    mom = (f*weights).sum()/np.sqrt(np.pi)
    return mom 

def GaussHermiteQuadrature(f,mu=0,sigma=1,Nquad=11):
    '''
    Compute the Gauss-Hermite Quadrature Integral of f. 
    f: function handle accepting one input which is normally distributed with mean mu and std sigma. 
    '''
    nodes,weights = hermgauss(Nquad)
    xvalues = np.sqrt(2)*sigma*nodes + mu 
    fvec = np.array([f(x) for x in xvalues])
    return (fvec*weights).sum()/np.sqrt(np.pi)

def cheby_coeff(n,m):
    coeffcheb = np.zeros(n)
    coeffcheb[n-1] = 1
    cheb = Chebyshev(coeffcheb)
    coef = cheb2poly(cheb.coef)
    return coef[m]

def PreComputationLogNormalIntegrals(sigE,order,basefunc='polynomial'):
    '''
    Let F be a function of X and Z=exp(z), where

    z' = rhoZ * z + e, 

    where e~N(0,sigE^2). Suppose that we are trying to approximate F by a linear combination 
    of polynomials in X and exp(z), i.e 

    Fhat = np.dot(b , Poly(X,z) )
    
    In economic models, one might eg approximate the value function V(k,z) in a NGM as a 
    linear combination of ordinary polynomials in k,z; to second order, 
    
    V(k,z) = b0 + b1*k + b2*exp(z) + b3*(k**2) + b4*(exp(z)**2) + b5*k*exp(z)
    
    In this case, note that 
    
    E(Fhat(X',Z')|Z) = np.dot( np.dot(b(x,exp(rhoZ*z)), E(exp(e)) )

    due to the linearity of the expectations operator. Precomputation recognizes that E(exp(e)) can be
    solved for *analytically* if the e's are normally distributed, or numerically if the z's are not 
    normal. 

    '''

    if basefunc == 'polynomial':
        '''
        PreCompCoeffs[i] is the precomputed expectation for any terms in which z is raised to a power i.
        '''
        PreCompCoeffs = np.array([np.exp( 0.5 * ((i * sigE)**2) ) for i in np.arange(order+1)])

    if basefunc == 'chebyshev':
        '''
        PreCompCoeffs[i,j] is the precomputed expectation for any terms in a Tensor grid containing the 
        jth Chebyshev Polynomial in z in which z is raised to the power i. 
        '''
        PreCompCoeffs_power = np.array([np.exp( 0.5 * ((i * sigE)**2) ) for i in np.arange(order+1)])
        PreCompCoeffs_cheby = np.array([cheby_coeff(order+1,i) for i in np.arange(order+1)])
        PreCompCoeffs = np.zeros((order+1,order+1))
        for i in range(order+1):
            for j in range(order+1):
                PreCompCoeffs[i,j] = PreCompCoeffs_cheby[j] * PreCompCoeffs_power[i]

    return PreCompCoeffs

# -----------------------------------------------------------------------------------------------------------
# Pareto Distribution Moments (JIT-compatible)
# -----------------------------------------------------------------------------------------------------------
@njit 
def Pareto_survival(X,Xmin,TailIndex):
    '''
    Survival Function of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        surv = 1
    else:
        surv = (Xmin/X)**TailIndex
    return surv

@njit 
def Pareto_cdf(X,Xmin,TailIndex):
    '''
    CDF of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        cdf = 0
    else:
        cdf = 1 - ((Xmin/X)**TailIndex)
    return cdf

@njit 
def Pareto_pdf(X,Xmin,TailIndex):
    '''
    PDF of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        pdf = 0
    else:
        pdf = TailIndex * (Xmin**TailIndex) * (X**(-(TailIndex+1)))
    return pdf

@njit 
def Pareto_CondExp_Int_Above(X,Xmin,TailIndex):
    '''
    Calculate int_X^infty Z f(Z) dZ where f(Z) is the Pareto distribution with (Location,Shape) = (Xmin,TailIndex)
    IMPORTANT: need TailIndex > 1
    '''
    if X < Xmin: 
        I = (TailIndex/(TailIndex-1)) * Xmin 
    else:  
        I = (TailIndex/(TailIndex-1)) * (Xmin ** TailIndex) * (X ** (1-TailIndex))
    return I

@njit
def Pareto_CondExp_Int_Below(X,Xmin,TailIndex):
    '''
    Calculate int_Xmin^Xbar Z f(Z) dZ where f(Z) is the Pareto distribution with (Location,Shape) = (Xmin,TailIndex)
    IMPORTANT: need TailIndex > 1
    '''
    if X < Xmin: 
        I = 0   # Technically not a well-defined object
    else:
        I = (TailIndex/(TailIndex-1)) * (Xmin ** TailIndex) * ((Xmin**(1-TailIndex)) - (X ** (1-TailIndex)))
    return I

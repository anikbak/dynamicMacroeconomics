# This set of routines estimates the loglikelihood of a linear state-space model 
# Follows Bocola (20xx) teaching notes for DSGE model estimation. 
import numpy as np
from time_series.signal_extraction import KalmanFilter
from scipy.stats import beta,gamma,chi2,expon,invgamma

# ===============================================================================
# Priors
# ===============================================================================

def prior_expon(x,param=1):
    '''
    param = lambda, where pdf = lambda * exp(-lambda x)
    so that E(x) = 1/lambda
    '''
    ex = expon(param)
    return ex.pdf(x)

def prior_gamma(x,param=1):
    '''
    pdf = ( x ** (alpha-1) ) * ( exp(-x)) / Beta(alpha,beta)
    params[0] = alpha
    params[1] = beta
    '''
    g = gamma(param)
    return g.pdf(x)

def prior_cauchy(x):
    return (1/np.pi) * (1/(1 + (x**2)))

def prior_chi2(x,param):
    '''
    param = df of distribution
    '''
    x2 = chi2(param)
    return x2.pdf(x)

def prior_invgamma(x,param=1):
    ig = invgamma(param)
    return ig.pdf(x)

def prior_beta(x,params):
    '''
    pdf = ( x ** (alpha-1) ) * ( (1-x) ** (beta-1) )/ Beta(alpha,beta)
    params[0] = alpha
    params[1] = beta
    '''
    b = beta(params[0],params[1])
    return b.pdf(x)

def prior_normal(x,params=[0,1]):
    z = (x-params[0])/params[1]
    return ((np.sqrt(2*np.pi) * params[1])**(-1)) * np.exp(-0.5*(z**2))

def prior_uniform(params=[0,1]):
    '''
    PDF of uniform distribution
    '''
    return 1/(params[1]-params[0])

def priors(values,prior_dist_list,prior_dist_params):
    '''
    computes priors for a Bayesian estimation under the assumption that all parameters are independent.

    values                  : Nx1 vector of values for parameters
    prior_dist_list         : list containing desired prior distributions for parameter i
    prior_dist_params       : dictionary containing parameters of priors

    For now, this code only accepts the priors listed above. To be edited. 

    '''
    # number of variables
    nvar = values.size
    prior_vector = np.zeros(nvar)
    for i in range(nvar):
        if prior_dist_list[i] == 'expon':
            prior_vector[i] = prior_expon(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'gamma':
            prior_vector[i] = prior_gamma(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'invgamma':
            prior_vector[i] = prior_invgamma(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'cauchy':
            prior_vector[i] = prior_cauchy(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'chi2':
            prior_vector[i] = prior_chi2(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'beta':
            prior_vector[i] = prior_beta(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'normal':
            prior_vector[i] = prior_normal(values[i],prior_dist_params[i])
        elif prior_dist_list[i] == 'uniform':
            prior_vector[i] = prior_uniform(values[i],prior_dist_params[i])
        else:
            print(f'Your desired prior for parameter {i}, {prior_dist_list[i]}, is not supported.')
            prior_vector[i] = 1

    prior_product = prior_vector.prod() 
    log_prior = (np.log(prior_vector)).sum()
    return prior_vector, prior_product, log_prior

# ===============================================================================
# Routines to compute the Linearized RBC Model's state-space form
# ===============================================================================
'''
Let y0,c0,i0,k0,n0,z0 be the log values of all variables in the deterministic 
steady state. 

Let y,c,i,k,n,z be log deviations from the steady state, and let Y,C,I,K,N,Z be 
the corresponding levels. 

The nonlinear RBC Model is 

params: palpha,pbeta,psigma,ppsi,N_ss,prhoz,psigz,prhoa,psiga
endogs: Y,C,I,K,N,Z,pchiN
states: Z(-1),K(-1)
shocks: e,a (TFP and Preferences) 

Z * (1-palpha) * ((K/N)**(palpha)) * C ** (-psigma) = pchin * exp(a) * (N ** ppsi) 
C ** (-psigma) = pbeta * (1 - pdelta + Z*palpha*((K/N)**(palpha-1))) * C(+1) ** (-psigma)
Y = Z * (K ** palpha) * (N ** (1-palpha))
Y = C + I
K = K(-1)*(1-pdelta) + I
Z = Z(-1)**prhoz * exp(e), e ~ N(0,sigz)
N_ss given
'''

def RealBusinessCycleModelSteadyState(params):
    '''
    A simple real business cycle model. 
    params: palpha,pbeta,psigma,ppsi,pchin,prhoz,psigz,prhoa,psiga
    endogs: Y,C,I,K,N,Z
    states: Z(-1),K(-1)
    shocks: e,a (TFP and Preferences (leisure)) 
    '''
    palpha,pbeta,psigma,pdelta,ppsi,N_ss,prhoz,psigz,prhoa,psiga = [i for i in params]
    
    # ============================
    # Steady State: 
    # ============================
    
    # all shocks set to zero
    e0 = 0 
    a0 = 0 
    z0 = 0 

    # solve equilibrium conditions
    r0 = (1/pbeta)-1
    MPK0 = r0 + pdelta 
    KbyN0 = MPK0 ** (1/(palpha-1))
    k0 = np.log(KbyN0) + np.log(N_ss) 
    i0 = np.log(pdelta) + np.log(k0)
    y0 = palpha * k0 + (1-palpha) * N_ss 
    c0 = np.log(np.exp(y0) - np.exp(i0))
    n0 = np.log(N_ss)
    MPN0 = (1-palpha) * np.exp(y0-n0)
    pchin = c0**(-psigma) * MPN0 / (N_ss ** (1/ppsi))
    w0 = np.log(MPN0)
    return y0,c0,i0,k0,n0,z0,e0,a0,pchin,r0,w0

def RealBusinessCycleModelLinearizedMatrices(params):
    '''
    Return the State-Space model associated with the simple linearized RBC Model
    '''
    

# ===============================================================================
# Main Routines for Bayesian Estimation
# ===============================================================================
def ObjectiveFunction(param_estimate,param_fixed,prior_dist_list,prior_dist_params,data):
    '''
    To document: how param_estimate,param_fixed,data are arranged
    '''
    LogLikelihood = ModelLikelihood(param_estimate,param_fixed,data)
    _,_,LogPrior = priors(param_estimate,prior_dist_list,prior_dist_params)
    return - (LogPrior + LogLikelihood)


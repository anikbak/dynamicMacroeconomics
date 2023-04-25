###################################################################################################
# Routines for Signal Extraction
###################################################################################################

import numpy as np
from scipy.stats import norm,multivariate_normal as mvn_scipy 
from scipy.linalg import solve_discrete_lyapunov
from numpy.random import normal,multivariate_normal as mvn_np

# One-dimensional signal extraction 

def SignalSimulation_1D(N,T,InitState,InitVariance,phi,sig_e,sig_v,seed_init=0):
    
    # Seed
    np.random.seed(seed_init)

    # Draw Shocks 
    e, v = normal(scale=sig_e,size=(T,N)),normal(scale=sig_v,size=(T,N))

    # Series
    States,Observations = np.zeros((T,N)),np.zeros((T,N))
    States_Init = normal(loc=InitState,scale=np.sqrt(InitVariance),size=(N))
    States[0] = phi * States_Init + v[0]
    Observations[0] = States[0] + e[0]

    for t in range(1,T):
        States[t] = phi * States[t-1] + v[t]
        Observations[t] = States[t] + e[t] 

    return Observations,States,e,v

def SignalExtractionProblem_1D(InitState,InitVariance,Observations,phi,sig_e,sig_v):
    '''
    Consider the signal extraction Problem
    y = z + sig_e * e
    z = phi * z(-1) + sig_v * v 

    e,v ~ N(0,1), N(0,1)

    A standard Kalman-Filter approach to this produces the following objects for the underlying state z.

    * OneStepCondMeans[t] = E(z_{t+1} | Y^t)
    * OneStepVariances[t] = E(var(z_{t+1}) | Y^t)

    where Y^t is the history of observations up to date t. 

    The likelihood function can be computed from the normality of the errors. We have, 
    L = PROD{ p(y_{t+1} | Y^t ) }
    where
    p(y_{t+1} | Y^t) = NormPDF( E(z_{t+1} | Y^t), P_{t+1,t} + sigma^2_e )
    
    '''
    # Length of Time Series
    T = Observations.size

    # Preallocate
    KalmanGains,OneStepCondMeans,OneStepVariances,Likelihood_vec = np.zeros(T),np.zeros(T),np.zeros(T),np.zeros(T)

    # First Period
    OneStepCondMeans[0] = InitState
    OneStepVariances[0] = InitVariance

    # Begin Iterations
    for t in range(T):

        # Calculate Likelihood 
        Ey,Vy = OneStepCondMeans[t],OneStepVariances[t] + sig_e**2
        Likelihood_vec[t] = norm.pdf(Ey,Vy)
        KalmanGains[t] = OneStepVariances[t]/(OneStepVariances[t] + (sig_e**2))
        tp = t+1
        if tp < T:

            # Kalman Filter
            OneStepCondMeans[tp] = phi * OneStepCondMeans[t] + phi * KalmanGains[t] * (Observations[t]-OneStepCondMeans[t])
            OneStepVariances[tp] = (phi ** 2) * OneStepVariances[t] - (phi ** 2) * KalmanGains[t] * OneStepVariances[t] + sig_v**2
    
    # Calculate log likelihood 
    LogLikelihood = np.log(Likelihood_vec).sum()

    return OneStepCondMeans,OneStepVariances,KalmanGains,Likelihood_vec,LogLikelihood 

# Multidimensional Filtering
def find_stationary_distribution_multivariateAR1(Phi,R,Sig):
    '''
    Consider the vector AR(1) process 

    z_t = Phi z_{t-1} + R * Eps_t

    where Eps_t ~ MVN(0,Sig)

    The stationary distribution of this process is multivariate normal with mean 0 and
    variance Lam given by the discrete Lyapunov Equation

    Lam = Phi Lam Phi' + R Sig R'
    '''

    Lam = solve_discrete_lyapunov(Phi,R @ Sig @ R.T)
    return Lam 

def KalmanFilter(N,T,Observations,Mu,A,H,Phi,R,Sig,InitState=[],InitVariance=[]):
    '''
    Consider a generalized linear filtering problem. The state-space model is 

    y_t = Mu + A * z_t + H * E_t 
    z_t = Phi z_{t-1} + R * Eps_t

    E_t ~ MVN(0,I)
    Eps_t ~ MVN(0,Sig)

    y_t are Observations (ky x T)
    Mu is (ky x 1)
    A is (ky x kz)
    H is (ky x ky)
    Phi, R are (kz x kz)
    
    InitState is (kz x 1)
    InitVariance is (kz x kz)

    '''

    # Initialize if necessary 
    if InitState == []:
        InitState = np.zeros(Phi.shape[0])
    if InitVariance == []:
        InitVariance = find_stationary_distribution_multivariateAR1(Phi,R,Sig)

    # Extract Data Dimensions and ensure compatibility
    ky,T = Observations.shape 
    kz = Phi.shape[0]

    print(f'Checking on Dimensions:')
    print(f'***********************')
    print(f'Observations is a {ky} x {T} matrix, implying data has {ky} measurement equations with observations made over {T} periods.')
    print(f'InitState is a {kz} x 1 matrix, implying that there are {kz} hidden state variables.')
    print(f'The process we are studying is ') 
    print(f'    ')
    print(f'    y_t = mu + A z_t + H E_t')
    print(f'    z_t = Phi * z_t-1 + R Eps_t')
    print(f'    ')
    print(f'This means that')
    print(f'    * Mu  should be {ky} x 1. It is actually {Mu.shape}')
    print(f'    * A   should be {ky} x {kz}. It is actually {A.shape}')
    print(f'    * H   should be {ky} x {ky}. It is actually {H.shape}')
    print(f'    * Phi should be {kz} x {kz}. It is actually {Phi.shape}')
    print(f'    * R   should be {kz} x {kz}. It is actually {R.shape}')
    print(f'    ')
    print(f' Further, InitState should be {kz} x 1; it is {InitState.shape}')
    print(f' and InitVariance should be {kz} x {kz}, and is actually {InitVariance.shape}')

    # Preallocate 
    dimK,dimV,dimE = (T,kz),(T,kz,kz),(T,)
    KalmanGains,OneStepCondMeans,OneStepVariances,Likelihood_vec = np.zeros(dimK),np.zeros(dimE),np.zeros(dimV),np.zeros(T)
    
    # First Period
    OneStepCondMeans[0] = InitState
    OneStepVariances[0] = InitVariance

    # Begin Iterations
    for t in range(T):

        # Calculate Likelihood 
        ydist_mean = Mu + A @ OneStepCondMeans[t]
        ydist_vcov = A @ OneStepVariances[t] @ A.T + H @ H.T 
        Likelihood_vec[t] = mvn_scipy(ydist_mean,ydist_vcov).pdf(Observations[t])

        # Calculate Kalman Gain Matrix 
        KalmanGains[t] = A @ (OneStepVariances[t].T) @ np.linalg.inv(ydist_vcov)
        v = Observations[t] - ydist_mean

        # Update One step
        tp = t + 1
        if tp < T: 

            # Kalman Filter 
            OneStepCondMeans[tp] = Phi @ (OneStepCondMeans[t] + KalmanGains[t] @ v)
            Ptt = OneStepVariances[t] - KalmanGains[t] @ A @ OneStepVariances[t].T
            OneStepVariances[tp] = Phi @ Ptt @ Phi.T + R @ Sig @ R.T
    
    # Calculate log likelihood 
    LogLikelihood = np.log(Likelihood_vec).sum()

    return OneStepCondMeans,OneStepVariances,KalmanGains,Likelihood_vec,LogLikelihood 
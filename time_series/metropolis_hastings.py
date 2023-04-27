# Metropolis-Hastings Algorithm to sample from unknown distribution

import numpy as np,matplotlib.pyplot as plt
from numpy.random import normal,uniform
from scipy.stats import norm

def Metropolis_Hastings_quniform(p,a=1,N=10000,init=1):
    '''
    p: a function handle for the distribution function we want to sample from. 
        p must accept only one argument: an input. 
    N = # draws   
    '''
    unifs = uniform(0,1,size=N)
    simulation = np.zeros(N)
    for i in range(N):
        if i % 500 == 0:
            print(i)
        if i == 0:
            simulation[i] = init  
        else:
            # draw from proposal
            proposal = uniform(simulation[i-1]-a,simulation[i-1]+a)
            acceptance_prob = np.minimum(1,p(proposal)/p(simulation[i-1]))
            if unifs[i]<=acceptance_prob:
                simulation[i] = proposal 
            else:
                simulation[i] = simulation[i-1]
    return simulation

def Metropolis_Hastings_qnormal(p,proposal_scale,init,N=10000):
    '''
    p           : Likelihood function
    N           : number of draws
    init        : initial value of chain
    '''
    unifs = uniform(0,1,size=N)
    simulation = np.zeros(N)
    for i in range(N):
        if i % 500 == 0:
            print(i)
        if i == 0:
            simulation[i] = init 
        else:
            # draw from proposal
            proposal_loc = simulation[i-1]
            proposal = normal(loc=proposal_loc,scale=proposal_scale)
            proposal_dist_1 = norm(loc=proposal_loc,scale=proposal_scale)
            proposal_dist_2 = norm(loc=simulation[i-1],scale=proposal_scale)
            ratio_NR = p(proposal)/p(simulation[i-1])
            ratio_DR = proposal_dist_1.pdf(proposal)/proposal_dist_2.pdf(simulation[i-1])
            acceptance_prob = np.minimum(1,ratio_NR/ratio_DR)
            if unifs[i] < acceptance_prob:
                simulation[i] = proposal 
            else:
                simulation[i] = simulation[i-1]
    return simulation


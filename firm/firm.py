########################################################################################################################
# ROUTINES TO ITERATE A DISTRIBUTION
########################################################################################################################

import numpy as np 
from numba import njit,cfunc

# ------------------------------------------------------------------------------------------------------------
# Some Production Functions and Related Objects 
# ------------------------------------------------------------------------------------------------------------
def CESoutput(x1,x2,weight1,weight2,elasticity):
    '''
    Finds output Y from 
    Y = ( weight1 * (x1**exponent) + weight2 * (x2**exponent) )**(1/exponent)
    where
    exponent = (elasticity-1)/elasticity
    
    x1,x2: Inputs, conformable np.ndarrays
    weight1,weight2: coefficients in front of x1,x2 respectively, np.floats
    elasticity: elasticity of subst. betweeen x1,x2
    
    '''
    exponent = (elasticity-1)/elasticity
    t1 = weight1 * (x1**exponent)
    t2 = weight2 * (x2**exponent)
    return (t1+t2)**(1/exponent)

def CESprice(p1,p2,weight1,weight2,elasticity):
    t1 = (weight1 ** elasticity) * (p1 ** (1-elasticity))
    t2 = (weight2 ** elasticity) * (p2 ** (1-elasticity))
    return (t1+t2)**(1/(1-elasticity))

def CESMargProd(x1,x2,weight1,weight2,elasticity):
    '''
    Finds marginal products of factors x1,x2 for the production function 
    Y = ( weight1 * (x1**exponent) + weight2 * (x2**exponent) )**(1/exponent)
    where
    exponent = (elasticity-1)/elasticity
    
    x1,x2: Inputs, conformable np.ndarrays
    weight1,weight2: coefficients in front of x1,x2 respectively, np.floats
    elasticity: elasticity of subst. betweeen x1,x2
    
    '''
    Y = CESoutput(x1,x2,weight1,weight2,elasticity)
    common = Y**(1/elasticity)
    MPK1 = common * (x1 ** (-1/elasticity))
    MPK2 = common * (x2 ** (-1/elasticity))
    return MPK1,MPK2

def CobbDouglasOutput(xvec,exponentvec):
    '''
    Compute outputs for a Cobb-Douglas Production Function
    '''
    Y = np.cumprod(xvec**exponentvec)
    return Y 

def CobbDouglasMarginalProducts(xvec,exponentvec):
    '''
    Compute marginal products for each factor in a Cobb-Douglas Production Function
    '''
    Y = CobbDouglasOutput(xvec,exponentvec)
    MargProds = [exponentvec[i]*Y/xvec[i] for i in range(len(xvec))]
    return MargProds

def CobbDouglasTwoFactorProfits(w,K,Z,palphaK,palphaN):
    '''
    Common element in many firm dynamics models: a static Cobb-Doug. block that 
        takes capital, TFP as given (predetermined)
        allows firm to choose labor optimally conditional on this capital

    This problem can be solved in closed form. Requires palphaK + palphaN <= 1.
    '''

    # Step 1: Labor Demand
    N =  ( w * (K**(-palphaK))/(Z*palphaN) ) **(1/(palphaN-1) )

    # Step 2: Output
    Y = Z * (K**palphaK) * (N**palphaN)

    # Step 3: Static Earnings/Profits
    Pi = Y - w * N

    return N,Y,Pi

def KORVFactorDemands(ws,wu,rk,Z,KORVlam,KORVmu,KORVsig,KORVrho,Bs=1,Bu=1,Bk=1):
    '''
    Given a production function like KORV (2000), calculate labor demands by directly inverting the cost function.
    parameters KORVlam, KORVmu, KORVsig and KORVrho are the parameters lambda, mu, sigma and rho in the original
    KORV (2000) notation. The Bs,Bu,Bk terms account for possible factor-augmenting technologies. 

    This code returns the factor ratios only.
    '''

    # Price Indices
    esig = 1/(1-KORVsig) 
    erho = 1/(1-KORVrho)
    pG = CESprice(ws/Bs,rk/Bk,1-KORVlam,KORVlam,erho)
    pY = CESprice(wu/Bu,pG,KORVmu,1-KORVmu,esig)/Z

    # Demand Curves assuming perfect competition
    LSbyK = (Bk/Bs)*(((ws/rk)/(Bs/Bk))**(-KORVrho))*(1-KORVlam)/KORVlam
    LUbyK = (Bk/Bu)*((wu/pG)**(-KORVsig))*((pG/rk)**(-KORVrho))*KORVmu/(1-KORVmu)

    return LSbyK,LUbyK

def IdealPriceIndex(prices,Distribution,elasticity):
    '''
    Input into models of firm dynamics. Given a distribution of prices over a state space, compute the ideal
    price index associated with a final goods retailer packaging outputs according to a CES aggregator.
    '''
    price_power = prices ** (1-elasticity)
    P = (Distribution * price_power).sum() ** (1/(1-elasticity))
    return P

# ------------------------------------------------------------------------------------------------------------
# TO ADD: Routines for Hopenhayn/Rogerson style models 
# ------------------------------------------------------------------------------------------------------------

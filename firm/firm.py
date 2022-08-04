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
@njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:])') 
def DynamicInvestment_Discrete_Step(VNext,profit,ZTrans,discount,invest,AdjCost):
    '''
    Given a value function for next period, current profit function and Transition, One VFI Step
    ********************************************************************************************
    One step of a VFI algorithm to solve the problem 
    V(k,Z) = max_k' pi(k,Z) + qk * ((1-pdelta)*k - k') - CapAdjCost(k',k) + discount*E [V(k',Z') | Z]

    VNext: np.ndarray((nZ,nK)), next period's value function
    profit: np.ndarray((nZ,nK)), corresponds to pi(k,Z), solved static profits substituting for statically optimally chosen inputs. 
    invest: np.ndarray((nK,nK)), corresponds to qk * ((1-pdelta)*k - k')
    AdjCost: np.ndarray((nK,nK)), Adjustment costs to go from kgrid[j] -> kgrid[i]
    ZTrans: np.ndarray((nZ,nZ)), Transition matrix

    '''

    # Preallocate
    V = np.zeros(VNext.shape)
    nZ,nK = V.shape

    # Expected Value Next Period 
    EVNext = discount*(ZTrans @ VNext)

    # Iteration
    for iZ in range(nZ):
        
        # Objective function to be maximized over vertical axis
        objective = np.tile(profit[iZ],(nK,1)) - invest - AdjCost + np.tile(EVNext[iZ][:,np.newaxis],(1,nK))
        
        # Perform Maximization
        V[iZ] = objective.max(0)

    return V 

def DynamicInvestmentCobbDouglas_Discrete(w,qk,k,Z,ZTrans,palphaK,palphaN,pdelta,discount,costParams,costSpec='nonconvex'):
    '''
    Given factor prices and grids, solve a dynamic firm problem. 
    ************************************************************
    Solve the problem 
    
    V(k,Z) = max_k' pi(k,Z) + qk * ((1-pdelta)*k - k') - CapAdjCost(k',k) + discount*E [V(k',Z') | Z]

    where
    
    pi(k,Z) = max_n Z k^palphaK n^{1-palphaN} - w*N

    w, rk: float64,float64, real wages, interest rates
    k,Z: ndarray((nZ,nk)),ndarray((nZ,nk)), tfp, capital grids. 
    kDense,ZDense: ndarray((nZ,nk)),ndarray((nZ,nk))
    '''

    # Step 0: Extract sizes and define some objects
    nZ,nK = k.shape
    ktoday = np.tile(k[0],(nK,1))
    knext = np.tile(k[0][:,np.newaxis],(1,nK))
    inv = qk*(knext-((1-pdelta)*ktoday))
    if costSpec == 'convex':
        AdjCost = (costParams[0]/2)*((inv/ktoday)**2)
    elif costSpec == 'nonconvex':
        AdjCost = (costParams[0]/2)*((inv/ktoday)**2) + costParams[1] * (inv != 0)
    elif costSpec == 'nonconvex_asymmetric':
        AdjCost = (costParams[0]/2)*((inv/ktoday)**2) + costParams[1] * (inv < 0) + costParams[2] * (inv > 0)

    # Step 1: Setup Profit Function on box for regular state
    n =  ( w * (k**(-palphaK))/(Z*palphaN) ) **(1/(palphaN-1) )
    y = Z * (k**palphaK) * (n**palphaN)
    pi = y - w * n

    # Step 2: Define One Iteration Step
    def Iteration(VNext): 
        # Preallocate
        V = np.zeros(VNext.shape)
        # Next period's expected value
        EVNext = discount * (ZTrans @ VNext)
        for iZ in range(nZ):
            # Objective function to be maximized over vertical axis
            objective = np.tile(pi[iZ],(nK,1)) - inv - AdjCost + np.tile(EVNext[iZ][:,np.newaxis],(1,nK))
            # Perform Maximization
            iMax = objective.argmax(0)
            V[iZ] = objective.max(0)
        return V 


    return 
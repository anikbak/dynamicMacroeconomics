##############################################################################################################
# Generic Routines
##############################################################################################################
import numpy as np 

def s2a(df,v,dim2=False,norm=False):
    if dim2 == False:
        if norm == True:
            return np.array(df[v]/df[v].iloc[0])
        if norm == False:
            return np.array(df[v])
    elif dim2 == True:
        if norm == True:
            return np.array(df[v]/df[v].iloc[0])[:,np.newaxis]
        if norm == False:
            return np.array(df[v])[:,np.newaxis]
    
# ------------------------------------------------------------------------------------------------------------
# Discrete Choice problems with Type 1 Extreme Value shocks
# Based on Auclert et al (2021)
# ------------------------------------------------------------------------------------------------------------
def logit_choice_probabilities(V,param_scale):
    '''
    Finds the Logit Choice probabilities along axis 0
    (i.e. axis 0 must index the options, so that V[i],V[j] are the values associated with
    choices i,j, etc. V[i] might itself be a state-dependent multidimensional object.)
    '''
    V_norm = V - V.max(axis=0)
    Vexp = np.exp(V_norm/param_scale)
    P = Vexp/Vexp.sum(axis=0)
    return P 

def logit_choice(V,param_scale):
    '''
    Logit Choice probabilities along 0th axis.
    '''
    const = V.max(axis=0)
    Vnorm = V - const
    Vexp = np.exp(Vnorm/param_scale)
    VexpSum = Vexp.sum(axis=0)
    P = Vexp/VexpSum
    EV = const + param_scale * np.log(VexpSum)
    return P,EV

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

def CobbDouglasMPK(xvec,exponentvec):
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
    Given a production function like KORV (2000), calculate labor demands by directly inverting the cost function
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
# Dynamic Routines
# ------------------------------------------------------------------------------------------------------------

def GPSum(Term1,CommonRatio,M):
    '''
    Calculate the sum of M terms of a GP starting at Term1 with common ratio CommonRatio.
    '''
    return Term1 * (1-(CommonRatio**M))/(1-CommonRatio)

def PresentValue(flows,rates):
    '''
    Let flows be a vector of cash flows starting at date t. That is, flows[i] = flows_{t+i}, i = 0,1,..., len(flows)-1
    Let rates be a vector of discount rates such that rates[i] is the discount rate between periods t+i-1 and t+i, 
    with rates[0] = 1. This corresponds to the "r_t" timing in macro, i.e. rates[i] = r_t+i
    This function returns the present value of flows discounted by rates. That is, it calculates

    sum_(i = 0)^(len(flows)-1) flows_(t+i) / prod_(j=0)^(i) (1+rates(t+j))

    where it is understood that prod_(j=0)^(i=0) (1+rates(t+j)) = 1. That is, for flows starting at t, the PV of the stream
    as of date t omits discounting by r_t. 

    rates: np.ndarray(T)
    flows: np.ndarray(T)
    '''
    rates = rates + 1
    rates[0] = 1
    R = rates.cumprod()
    return (flows/R).sum()

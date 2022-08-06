########################################################################################################################
# ROUTINES TO ITERATE A DISTRIBUTION
########################################################################################################################
import numpy as np 
from numba import njit,cfunc
from numba.types import UniTuple,Tuple
from routines.grids import StateSpaces
from distribution.distribution import DistributionStep1D

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
# Routines for Dynamic Investment Models 
# ------------------------------------------------------------------------------------------------------------
@njit('Tuple([f8[:,:],u4[:,:]])(f8[:,:],f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:])') 
def DynamicInvestment_Discrete_Step(VNext,profit,ZTrans,discount,invest,AdjCost):
    '''
    Given a value function for next period, current profit function and Transition, One VFI Step
    ********************************************************************************************
    One step of a VFI algorithm to solve the problem 
    V(Z,k) = max_k' pi(Z,k) + qk * ((1-pdelta)*k - k') - CapAdjCost(k',k) + discount*E [V(Z',k') | Z]

    VNext: np.ndarray((nZ,nK)), next period's value function
    profit: np.ndarray((nZ,nK)), corresponds to pi(k,Z), solved static profits substituting for statically optimally chosen inputs. 
    invest: np.ndarray((nK,nK)), corresponds to qk * ((1-pdelta)*k - k')
    AdjCost: np.ndarray((nK,nK)), Adjustment costs to go from kgrid[j] -> kgrid[i]
    ZTrans: np.ndarray((nZ,nZ)), Transition matrix

    '''

    # Preallocate
    V,ix = np.zeros(VNext.shape,dtype='float64'),np.zeros(VNext.shape,dtype='uint32')
    nZ,nK = V.shape

    # Expected Value Next Period 
    EVNext = discount*(ZTrans @ VNext)

    # Iteration
    for iZ in range(nZ):
        
        # Objective function to be maximized over vertical axis
        objective = profit[iZ] - invest - AdjCost + EVNext[iZ][:,np.newaxis]
        objective = objective.T 

        # Perform Maximization
        for iK in range(nK):
            ix[iZ,iK] = objective[iK].argmax()
            V[iZ,iK] = objective[iK].max()

    return V,ix 

@njit('f8[:,:](f8[:,:],u4[:,:],f8[:,:],f8[:,:],f8,f8[:,:],f8[:,:])')
def DynamicInvestment_Discrete_HowardStep(VNext,ix,profit,ZTrans,discount,invest,AdjCost):
    '''
    One Howard Iteration Step. 
    ********************************************************************************************
    VNext: np.ndarray((nZ,nK)), next period's value function
    ix: np.ndarray((nZ,nK)), policy function st ix[i,j] is the index of k'(Z[i],k[j]) on the grid for k 
    profit: np.ndarray((nZ,nK)), corresponds to pi(Z,k), solved static profits substituting for statically optimally chosen inputs. 
    invest: np.ndarray((nK,nK)), corresponds to qk * ((1-pdelta)*k - k')
    AdjCost: np.ndarray((nK,nK)), Adjustment costs to go from kgrid[j] -> kgrid[i]
    ZTrans: np.ndarray((nZ,nZ)), Transition matrix
    '''

    # Preallocate
    V = np.zeros(VNext.shape,dtype='float64')
    nZ,nK = V.shape

    # Expected Value Next Period 
    EVNext = discount*(ZTrans @ VNext)

    # Iteration
    for iZ in range(nZ):
        for iK in range(nK):
            idNext = ix[iZ,iK]
            V[iZ,iK] = - invest[idNext,iK] - AdjCost[idNext,iK] + EVNext[iZ,idNext]
    
    V = V + profit 
    return V

def DynamicInvestmentCobbDouglas_Discrete_SteadyState(w,qk,k,Z,ZTrans,palphaK,palphaN,pdelta,discount,costParams,costSpec='convex',nHoward=10,maxit=1000,tol=1e-7,noisily=10):
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
    #ktoday = np.tile(k[0],(nK,1))
    ktoday = k[0]
    invest = qk*(k[0][:,np.newaxis]-(1-pdelta)*k[0])

    if costSpec == 'convex':
        AdjCost = (costParams[0]/2)*((invest/ktoday)**2)
    elif costSpec == 'nonconvex':
        AdjCost = (costParams[0]/2)*((invest/ktoday)**2) + costParams[1] * (invest != 0)
    elif costSpec == 'nonconvex_asymmetric':
        AdjCost = (costParams[0]/2)*((invest/ktoday)**2) + costParams[1] * (invest < 0) + costParams[2] * (invest > 0)

    # Step 1: Setup Profit Function and investment 
    n =  ( w * (k**(-palphaK))/(Z*palphaN) ) **(1/(palphaN-1) )
    y = Z * (k**palphaK) * (n**palphaN)
    pi = y - w * n

    # Step 2: Iterations
    V,ix = np.zeros(k.shape,dtype='float64'),np.zeros(k.shape,dtype='uint32')
    it,err = 0,1
    while (err>=tol) & (it<=maxit):
        
        # Initial Value 
        V0 = V.copy() 
        
        # Update Iteration
        V,ix = DynamicInvestment_Discrete_Step(V0,pi,ZTrans,discount,invest,AdjCost)
        while (err>=tol) & (iH <= nHoward):
    
            # Howard Step 
            VH = V.copy()
            V = DynamicInvestment_Discrete_HowardStep(VH,ix,pi,ZTrans,discount,invest,AdjCost)
            err = np.max(np.abs(V-V0)/(V+V0+tol))
            iH += 1
        
        # Report 
        if noisily != 0:
            if it % noisily == 0:
                print(f'iteration {it}, error {err}')
        it += 1
    
    return V,ix

# ------------------------------------------------------------------------------------------------------------
# TO ADD: Routines for Hopenhayn/Rogerson style models 
# ------------------------------------------------------------------------------------------------------------
def HopenhaynRogersonCobbDouglas_SS(maxit=21,tol=1e-4,ptheta=100,discount=0.8,pEntryCost=40,pFixed=20,palphaN=0.67,ptau=0.1,prhoZ=0.9,psigZ=0.2,pmuZ=0,nZ=33,pmZ=3,lmin=1,lmax=1000,nl=1000):
    # ========================================================================================================================
    # Routine to find Equilibrium
    # ========================================================================================================================
    # 
    # Note that the numeraire is labor, so the wage = 1 identically. 
    # 
    # Note orientation of boxes is such that the maximum is taken over columns, which is important since numpy uses row-major
    # order. 
    # ========================================================================================================================

    # Step 1: Setup State Spaces
    lgrid,l,_,Z,ZTrans,_ = StateSpaces(['l'],['Z'],[np.log(lmin)],[np.log(lmax)],[nl],[prhoZ],[psigZ],[pmuZ],[nZ],[pmZ],['python'],['python'],['rouwenhorst'])
    lgrid = np.exp(lgrid['l'])
    l = np.exp(l['l'])
    Z = np.exp(Z['Z'])
    ZTrans = ZTrans['Z']
    ZStat = ZTrans.copy()  
    for i in range(100):
        ZStat = ZStat @ ZTrans
    ZStat = ZStat[0]
    EZ = np.dot(ZStat,Z[:,0])
    Z = np.exp(1.39) * Z/EZ

    # Step 2: Given State Spaces, construct adjustment cost box (essentially a firing cost) 
    AdjCost = ptau*np.maximum(0,lgrid[:,np.newaxis]-lgrid)
    ExitCost = ptau*lgrid

    # Step 2: Conjecture a Price 
    p_min,p_max = 1,2

    # Step 3: Iterations
    it,err = 0,1
    while (err>tol) & (np.abs(p_min-p_max) > 0.01*tol) & (it<=maxit):
        
        # Current Price 
        p = 0.5*(p_min+p_max)
        
        # Current profits
        pi = p * Z * (l ** palphaN) - l 

        # Inner iteration loop 
        itIn,errIn,maxitIn,tolIn = 0,1,500,tol*0.1 
        value = pi/(1-discount)
        ipolicy,policy_l,policy_x = np.empty(value.shape,dtype='uint32'),np.empty(value.shape),np.zeros(value.shape,dtype='uint32')

        while (errIn>tolIn) & (itIn<=maxitIn):

            value0 = value.copy()

            # Expected Value next period
            EVNext = discount * (ZTrans @ value0)

            # Iterate 
            for iZ in range(nZ):
                
                # Objective function for firm this period if it chooses not to exit 
                objective = EVNext[iZ] + pi[iZ][:,np.newaxis] - AdjCost - pFixed

                # Value for firm if it chooses not to exit 
                ipolicy[iZ] = objective.argmax(0)
                valueCont = np.array([objective[i,ipolicy[iZ,i]] for i in range(nl)])

                # Assign policy for firm
                policy_x[iZ] = valueCont < -ExitCost
                idCont = np.where(policy_x[iZ]==0)[0]
                
                value[iZ,policy_x[iZ]] = -ExitCost
                policy_l[iZ,policy_x[iZ]] = 0

                for i in idCont:
                    value[iZ,i] = valueCont[i]
                    policy_l[iZ,i] = lgrid[ipolicy[iZ,i]]

            # Update criterion 
            errIn = np.max(np.abs(value-value0)/(np.abs(value0)+1))
            itIn += 1

        # Compute value of entry
        VEntry = discount * np.dot(value[:,0],ZStat)
        if VEntry > pEntryCost:
            p_max = p 
        elif VEntry < pEntryCost:
            p_min = p 

        # Update criterion
        err = 2*np.abs(VEntry-pEntryCost)/(1+np.abs(VEntry+pEntryCost))
        it += 1  

        # Display 
        print(f'Current iteration: ')
        print(f'convergence in value: {errIn}')
        print(f'range for p: {p_min,p_max}')
        print(f'convergence in entry: VEntry = {VEntry} vs EntryCost = {pEntryCost}')

    # Update policies
    policy_hire = policy_l > l 
    policy_fire = policy_l < l 
    policy_noch = policy_l == l 

    # Steady-State Distribution over state space 
    dist = np.tile(ZStat[:,np.newaxis],(1,nl))
    dist = dist/dist.sum() 
    itDist,errDist = 0,1 
    entry = np.zeros((nZ,nl))
    entry[:,0] = ZStat

    while (errDist > tol) & (itDist <= maxit):

        dist0 = dist.copy()
        dist = DistributionStep1D(dist0,policy_l,l,ZTrans,1,policy_x,entry)
        errDist = np.max(np.abs(dist-dist0))
        itDist += 1

    print(f'Distribution convergence: {errDist}')

    # Choose Mass of firms 
    y = (dist * Z * (l**palphaN)).sum()
    D = ptheta/p
    M = D/y.sum()

    return p,value,policy_l,policy_x,policy_hire,policy_fire,policy_noch,pi,dist,y,M,l,Z 

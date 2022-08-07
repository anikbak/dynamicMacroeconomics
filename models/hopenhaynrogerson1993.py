# =================================================================================================================================
# Routines to solve the Hopenhayn-Rogerson (1993) Model of firm dynamics with employment adjustment costs
# =================================================================================================================================

# Imports 
import numpy as np 
from routines.grids import StateSpaces,StationaryDistribution
from distribution.distribution import DistributionStep1D

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

    # Step 1: Setup State Spaces, Logarithmically-spaced for employment
    lgrid,l,_,Z,ZTrans,_ = StateSpaces(['l'],['Z'],[np.log(lmin)],[np.log(lmax)],[nl],[prhoZ],[psigZ],[pmuZ],[nZ],[pmZ],['python'],['python'],['rouwenhorst'])
    lgrid = np.exp(lgrid['l'])
    l = np.exp(l['l'])
    Z = np.exp(Z['Z'])

    # Stationary Distribution of the AR(1), and normalized values for the stochastic state
    ZTrans = ZTrans['Z']
    ZStat = StationaryDistribution(ZTrans)
    EZ = np.dot(ZStat,Z[:,0])
    Z = np.exp(1.39) * Z/EZ

    # Step 2: Given State Spaces, construct adjustment cost box (essentially a firing cost) 
    AdjCost = ptau*np.maximum(0,lgrid[:,np.newaxis]-lgrid)
    ExitCost = ptau*lgrid

    # Step 3: Conjecture a Price 
    p_min,p_max = 1,2

    # Step 4: Iterations
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

    # Step 5: Steady-State Distribution over state space 
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

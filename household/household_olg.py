#########################################################################################################################
# OVERLAPPING GENERATIONS MODELS
#########################################################################################################################
import numpy as np 
from numba import njit,guvectorize 

@guvectorize(['void(float64[:], float64[:], float64[:], float64[:])'], '(n),(nq),(n)->(nq)', nopython=True)
def interpolate_y(x, xq, y, yq):
    """Efficient linear interpolation exploiting monotonicity.

    Adapted from Adrien's notebook (Auclert et al 2021)

    Complexity O(n+nq), so most efficient when x and xq have comparable number of points.
    Extrapolates linearly when xq out of domain of x.

    Parameters
    ----------
    x  : array (n), ascending data points
    xq : array (nq), ascending query points
    y  : array (n), data points

    Returns
    ----------
    yq : array (nq), interpolated points
    """
    nxq, nx = xq.shape[0], x.shape[0]

    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in range(nxq):
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi_cur = (x_high - xq_cur) / (x_high - x_low)
        yq[xqi_cur] = xqpi_cur * y[xi] + (1 - xqpi_cur) * y[xi + 1]
    
@guvectorize(['void(float64[:], float64[:], uint32[:],uint32[:],float64[:])'], '(n),(nq)->(nq),(nq),(nq)', nopython=True)
def interpolate_coord(x, xq, xqi, xqia, xqpi):

    """
    Adapted from Adrien's notebook (Auclert et al 2021)
    Efficient linear interpolation. xq = xqpi * x[xqi] + (1-xqpi) * x[xqia]
    x must be sorted, but xq does not have to be

    Parameters
    ----------
    x    : array (n), ascending data points
    xq   : array (nq), query points
    xqi  : array (nq), empty to be filled with indices of lower bracketing gridpoints
    xqia  : array (nq), empty to be filled with indices of upper bracketing gridpoints
    xqpi : array (nq), empty to be filled with weights on lower bracketing gridpoints
    """

    # size of arrays
    nxq, nx = xq.shape[0], x.shape[0]

    # sort and keep track of initial order
    ind_new = np.argsort(xq)
    ind_init = np.argsort(ind_new)
    xq = xq[ind_new]

    # take care of values below and above minimum
    id = np.arange(nxq)[(x[0] <= xq) & (xq <= x[nx - 1])]
    xqi[xq < x[0]] = 0
    xqpi[xq < x[0]] = 1
    xqi[xq > x[nx - 1]] = nx - 1
    xqpi[xq > x[nx - 1]] = 1

    # interpolation
    xi = 0
    x_low = x[0]
    x_high = x[1]
    for xqi_cur in id:
        xq_cur = xq[xqi_cur]
        while xi < nx - 2:
            if x_high >= xq_cur:
                break
            xi += 1
            x_low = x_high
            x_high = x[xi + 1]

        xqpi[xqi_cur] = (x_high - xq_cur) / (x_high - x_low)
        xqi[xqi_cur] = xi

    # revert back to initial order
    xqpi[:] = xqpi[ind_init]
    xqi[:] = xqi[ind_init]

    # Compute index of point above, or same if last on the list
    xqia[:] = xqi[:] + 1
    xqia[xqia >= nx - 1] = nx - 1
    xqia[xq < x[0]] = xqi[xq < x[0]]

#########################################################################################################################
# ROUTINES TO SOLVE HOUSEHOLD PROBLEMS
#########################################################################################################################
def HouseholdOG_nonstochlifespan(assetGrid,incomeProcessGrid,incomeProcessTrans,pPhi = 1,pcrra=3,pdelta=0.03,r=0.04,nAge=80):
    '''
    ---------------------------------------------------------------------------------------------------------------------
    Routine to compute optimal household consumption in the one-asset model with constant interest rate
    ---------------------------------------------------------------------------------------------------------------------

    V(asset,incomeProcess,age)  = u(c) + pbeta * V(asset',incomeProcess',age+1), age <= nAge-1
                                = u(c) + v(asset') 

    subject to 

    asset' = (1+r)*asset + incomeProcess(age) - c
    incomeProcess' = Markov(incomeProcess) 
    
    where u(.) = CRRA utility with parameter pcrra and v(.) = Phi * u(.) (i.e. homothetic preferences)
    '''

    # Step 0: Preallocate 
    nZ = incomeProcessGrid.size 
    nA = assetGrid.size 
    income, asset = np.tile(incomeProcessGrid[:,np.newaxis],(1,nA)), np.tile(assetGrid,(nZ,1))
    consumption, saving, marginalUtility = np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA))

    # Step 1: Final Consumption 
    consumption[nAge-1] = ((1+r)*asset + income)/(1 + (pPhi**(-pcrra)))
    saving[nAge-1] = (pPhi**(-pcrra)) * ((1+r)*asset + income)/(1 + (pPhi**(-pcrra)))
    marginalUtility[nAge-1] = consumption[nAge-1]**(-pcrra)

    # Step 2: Use a version of EGM to iterate backward on Euler equation 
    for age in range(nAge-2,-1,-1):

        # One iteration on Euler Equation
        marginalUtility[age] =  ((1+r)/(1+pdelta)) * (incomeProcessTrans @ marginalUtility[age+1])
        cons_nextgrid = marginalUtility[age]**(-1/pcrra)

        # Today's cash on hand
        cash_on_hand = (1+r)*asset + income 

        # Current assets as a function of current cash on hand 
        for iZ in range(nZ):
            saving[age][iZ] = interpolate_y_cfunc(cons_nextgrid+asset[iZ],cash_on_hand[iZ],asset[iZ])

        saving[age] = np.maximum(saving[age],asset.min())
        consumption[age] = cash_on_hand - saving[age]

    return consumption, saving, marginalUtility

#########################################################################################################################
# ROUTINES TO SOLVE FOR A COHORT'S PROBLEM IN AN OG MODEL WITH RISK
#########################################################################################################################

def SSDist(DistInit,DistNext0,idL,idH,wL,wH,maxit=10000,tol=1e-5,noisily=True):
    dist = DistInit.copy() 
    it,err = 0,1
    while (err>=tol) & (it<=maxit):
        distInit = dist.copy() 
        dist = DistUpdate(distInit,DistNext0,idL,idH,wL,wH)
        err = np.max(np.abs(dist-distInit)/dist.sum())
        it += 1
        if noisily: 
            print(f'iter {it}, error {err}')
    return dist  

@njit 
def DistPath(DistNext0,incomeProcessTrans,idL,idH,wL,wH):
    '''
    DistNext0 : np.array((nZ,nA)), initial distribution at age 0 for all households
    Dist_Path : np.array((nAge,nZ,nA)), output distribution
    idL,idH,wL,wH: np.array((nAge,nZ,nA)), outcome of interpolating policy function onto dense grid
    '''

    Dist_Path = np.zeros(idL.shape)
    Dist_Path[0] = DistNext0
    nAge,nZ,nA = idL.shape
    for iage in range(nAge-1):
        for ia in range(nA):
            for iZ in range(nZ):
                Lnext,Hnext,wLnext,wHnext = idL[iage,iZ,ia],idH[iage,iZ,ia],wL[iage,iZ,ia],wH[iage,iZ,ia]
                Dist_Path[iage+1,iZ,Lnext] += wLnext * Dist_Path[iage,iZ,ia] 
                Dist_Path[iage+1,iZ,Hnext] += wHnext * Dist_Path[iage,iZ,ia]
        Dist_Path[iage+1] = incomeProcessTrans.T @ Dist_Path[iage+1]
    return Dist_Path 

def CohortProblem(wPath,rPath,discountPath,nonlabincPath,collegeCostPath,ageProfile,assetGrid,incomeProcessGrid,incomeProcessTrans,DistNext0,pcrra=1,pPhi=1):
    '''
    ***********************************************************************
    Cohort Problem for one-asset Overlapping Generations Labor Supply Block
    ***********************************************************************
    
    Wages, real interest rate, discount rate
    ****************************************
    wPath: np.ndarray(nAge)
    rPath: np.ndarray(nAge)
    discountPath: np.ndarray(nAge)

    Income Paths
    ************
    nonlabincPath: np.ndarray((nAge,nZ,nA))
    ageProfile: np.ndarray(nAge), age Profile of labor productivity 
    collegeCost: np.ndarray(nAge), equal to 0 at all ages >=4 but for ages 0,1,2,3 may be positive. 
    '''

    # Step 0: Construct Incomes at each date
    nZ,nA,nAge = incomeProcessGrid.size,assetGrid.size,ageProfile.size 
    income,asset = np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA))
    for iAge in range(nAge):
        income[iAge] = ageProfile[iAge] * wPath[iAge] * np.tile(incomeProcessGrid[:,np.newaxis],(1,nA)) + nonlabincPath[iAge] - collegeCostPath[iAge]
        asset[iAge] = np.tile(assetGrid,(nZ,1))

    # Step 1: Preallocate
    consumption, saving, marginalUtility,coh = np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA)),np.zeros((nAge,nZ,nA))

    # Step 2: Final Consumption
    coh[nAge-1] = (1+rPath[nAge-1])*asset + income[nAge-1]

    if pPhi != 0: 
        consumption[nAge-1] = (coh[nAge-1])/(1+(pPhi**(-pcrra)))
        saving[nAge-1] = coh[nAge-1] - consumption[nAge-1]
    
    else: 
        saving[nAge-1] = np.zeros((nZ,nA))
        consumption[nAge-1] = coh[nAge-1]

    marginalUtility[nAge-1] = consumption[nAge-1]**(-pcrra)

    # Step 3: Iterations
    for age in range(nAge-2,-1,-1):
        
        # One iteration on Euler Equation
        marginalUtility[age] =  ((1+rPath[age+1])/(1+discountPath[age+1])) * (incomeProcessTrans @ marginalUtility[age+1])
        cons_nextgrid = marginalUtility[age]**(-1/pcrra)

        # Today's cash on hand
        coh[age] = (1+rPath[age])*asset + income[age]
        coht = coh[age]

        # Current assets as a function of current cash on hand 
        assett = asset[iage]
        for iZ in range(nZ):
            saving[age][iZ] = interpolate_y(cons_nextgrid[iZ]+assett[iZ],coht[iZ],assett[iZ])

        saving[age] = np.maximum(saving[age],asset.min()-1e-5)
        consumption[age] = coht - saving[age]

    # Distribution of this cohort over state space
    idL,idH,wL = interpolate_coord(assetGrid,saving)
    wH = 1-wL
    distPath = DistPath(DistNext0,idL,idH,wL,wH)

    # Solve for Value
    idn0,idn1,idn2 = np.where(consumption<0)
    consumptionp = consumption 
    consumptionp[idn0,idn1,idn2] = 1e-9
    if pcrra == 1:
        utility = np.log(consumptionp)
    else:
        utility = (1/(1-pcrra)) * ((consumptionp**(1-pcrra))-1)
    Value = np.zeros(utility.shape)
    Value[nAge-1] = utility[nAge-1]
    for iage in range(nAge-2,-1,-1):
        for ia in range(nA):
            EVNext = np.zeros(nZ)
            for iz in range(nZ):
                EVNext[iz] = wL[iage,iz,ia]*Value[iage+1,iz,idL[iage,iz,ia]] + wH[iage,iz,ia]*Value[iage+1,iz,idH[iage,iz,ia]]
            Value[iage,:,ia] = utility[iage,:,ia] + (1/(1+discountPath[iage]))*(incomeProcessTrans @ EVNext)

    return Value,utility,consumption,saving,marginalUtility,coh,distPath,income,asset

@njit 
def IncomeInterpolator(policy,idL,idH,wL,wH):
    '''
    policy is a policy function with dimensions (nAge,nZ,nA). 
    idL,idH,wL,wH are interpolation indices and weights with dimension (nAge,nZ',nA). 
    '''
    nAge,nZp,nA = idL.shape
    policy_out = np.zeros((nAge,nZp,nA))
    idL = np.uint32(idL)
    idH = np.uint32(idH)
    for iAge in range(nAge):
        for iZ in range(nZp):
            for iA in range(nA):
                idxL,idxH = idL[iAge,iZ,iA],idH[iAge,iZ,iA]
                policy_out[iAge,iZ,iA] = wL[iAge,iZ,iA] * policy[iAge,idxL,iA]  + wH[iAge,iZ,iA] * policy[iAge,idxH,iA]
    return policy_out

def OG_Labor_Block_Discrete_Types(nTypes,wPaths,rPaths,discountPaths,nonlabincPaths,collegeCostPaths,ageProfiles,assetGrid,incomeProcessGrids,incomeProcessTranss,DistNext0s,pcrras,pPhis):
    '''
    *************************************************************************************************************************************
    Master Routine to Solve an OG Economy with a set of heterogeneous types of agents 
    *************************************************************************************************************************************

    nTypes: number of types of agents in the economy 

    Paths for Income Variables
    **************************
    wPaths: np.ndarray((nTypes,nAge))
    rPaths: np.ndarray((nTypes,nAge))
    discountPaths: np.ndarray((nTypes,nAge))
    nonlabincPaths: np.ndarray((nTypes,nAge))
    collegeCostPaths: np.ndarray((nTypes,nAge))
    ageProfiles: np.ndarray((nTypes,nAge))

    Grids
    *****
    Assumes that all grids have the same size, but possibly different values.
    assetGrid: np.ndarray(nA)
    incomeProcessGrids: np.ndarray((nTypes,nZ))
    incomeProcessTranss: np.ndarray((nTypes,nZ,nZ))
    DistNext0s: np.ndarray((nTypes,nA,nZ))
    pcrras: np.ndarray(nTypes)
    pPhis: np.ndarray(nTypes)

    *************************************************************************************************************************************
    '''
    # Solve for all the different types
    Value,utility,consumption,saving,marginalUtility,coh,distPath,income,asset = {},{},{},{},{},{},{},{},{}
    for iType in range(nTypes):
        Value[iType],utility[iType],consumption[iType],saving[iType],marginalUtility[iType],coh[iType],distPath[iType],income[iType],asset[iType] = CohortProblem(wPaths[iType],rPaths[iType],discountPaths[iType],nonlabincPaths[iType],collegeCostPaths[iType],ageProfiles[iType],assetGrid,incomeProcessGrids[iType],incomeProcessTranss[iType],DistNext0s[iType],pcrras[iType],pPhis[iType])
    
    # Combine all of them. To do this, observe that all types have a common asset grid but potentially different income grids. 
    # We need to interpolate the values across the income grid.
    # This step is going to be expensive.
    nAge,nZ,nA = Value[0].size
    nZp = nZ*2
    incomeGrid_master = np.linspace(incomeProcessGrids.min(),incomeProcessGrids.max(),nZp)
    incomeGrid_master_Box = np.zeros((nAge,nZ,nA))

    for iAge in range(nAge):
        incomeGrid_master_Box[iAge] = np.tile(incomeGrid_master[:,np.newaxis],(1,nA))
    
    for iType in range(nTypes):
        
        # Interpolation weights from initial income type to master box. Interpolate_coord is guvectorized so this operation is fast-ish.
        idLinc,idHinc,wLinc = interpolate_coord(incomeProcessGrids[iType],incomeGrid_master_Box)
        wHinc = 1-wLinc 

        # Perform interpolation
        Value[iType] = IncomeInterpolator(Value[iType],idLinc,idHinc,wLinc,wHinc)  
        utility[iType] = IncomeInterpolator(utility[iType],idLinc,idHinc,wLinc,wHinc) 
        consumption[iType] = IncomeInterpolator(consumption[iType],idLinc,idHinc,wLinc,wHinc) 
        saving[iType] = IncomeInterpolator(saving[iType] + saving[iType],idLinc,idHinc,wLinc,wHinc) 
        marginalUtility[iType] = IncomeInterpolator(marginalUtility[iType],idLinc,idHinc,wLinc,wHinc) 
        coh[iType] = IncomeInterpolator(coh[iType],idLinc,idHinc,wLinc,wHinc) 
        distPath[iType] = IncomeInterpolator(distPath[iType],idLinc,idHinc,wLinc,wHinc) 
        income[iType] = IncomeInterpolator(income[iType],idLinc,idHinc,wLinc,wHinc) 
        asset[iType] = IncomeInterpolator(asset[iType],idLinc,idHinc,wLinc,wHinc)

    # Construct Macroeconomic Aggregates from the Household Side of the Economy 
    Distribution = np.zeros((nAge,nZp,nA))
    Consumption = np.zeros((nAge,nZp,nA))
    Saving = np.zeros((nAge,nZp,nA))
    Asset = np.zeros((nAge,nZp,nA))

    for iType in range(nTypes):
        Distribution += distPath[iType]
        Consumption += consumption[iType] * distPath[iType]
        Saving += saving[iType] * distPath[iType]
        Asset += asset[iType] * distPath[iType]

    return Distribution,Consumption,Saving,Asset,Value,utility,consumption,saving,marginalUtility,coh,distPath,income,asset







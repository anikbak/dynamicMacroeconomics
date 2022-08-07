#########################################################################################################################
# OVERLAPPING GENERATIONS MODELS
#########################################################################################################################
import numpy as np 
from routines.interpolation import * 

#########################################################################################################################
# ROUTINES TO SOLVE HOUSEHOLD PROBLEMS
#########################################################################################################################
def HouseholdOG_nonstochlifespan(assetGrid,incomeProcessGrid,incomeProcessTrans,pPhi = 1,pcrra=3,pdelta=0.03,r=0.03,nAge=80):
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
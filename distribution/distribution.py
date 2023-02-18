########################################################################################################################
# ROUTINES TO ITERATE A DISTRIBUTION
########################################################################################################################
import numpy as np
from numba import njit 
from routines.interpolation import interpolate_coord_cfunc

@njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8,u4[:,:],f8[:,:])')
def DistributionStep1D(Dist,Policy,X,Trans,Mass,Exit,Entry):
    '''
    One Iteration of the Distribution Function given the current policy function. 
    *****************************************************************************
    Policy: np.ndarray((nZ,nX)), policy function evaluated on state space
    X: np.ndarray((nZ,nX)), nonstochastic state
    Z: np.ndarray((nZ,nX)), stochastic state
    Trans: np.ndarray((nZ,nZ)), transition matrix P such that P[i,j] = Pr(X' = X_j | X = X_i)
    Mass: np.float64, total mass of firms at period t+1
    Exit: np.uint32((nZ,nX)) of 1's at states at which agents exit the economy after end of date t and before start of date t+1
    Entry: np.float64((nZ,nX)) of *mass* of agents entering the economy after end of date t and before start of date t+1
    '''

    # Step 1: Project Policy onto state space 
    stateshape = Policy.shape
    nZ,_ = stateshape[0],stateshape[1]
    iL,iH,wL = np.zeros(stateshape,dtype=np.uint32),np.zeros(stateshape,dtype=np.uint32),np.zeros(stateshape,dtype=np.float64)
    for iZ in range(nZ):
        iL[iZ],iH[iZ],wL[iZ] = interpolate_coord_cfunc(X[iZ],Policy[iZ])
    wH = 1-wL 

    # Step 2: Account for Exit of agents
    Dist = Dist * (1-Exit)

    # Step 3: Use indices to assign non-exiting mass
    idx,idy = np.where((Dist/Dist.sum() >= 1e-11))
    N = idx.size
    DistNew = np.zeros(stateshape)

    for i in range(N):
        x,y = idx[i],idy[i]
        iLi,iHi = iL[x,y],iH[x,y]
        DistNew[x,iLi] += wL[x,y] * Dist[x,y] 
        DistNew[x,iHi] += wH[x,y] * Dist[x,y]
    
    DistNew = np.ascontiguousarray(Trans.T) @ np.ascontiguousarray(DistNew)

    # Entry 
    DistNew = DistNew + Entry
    return DistNew * Mass / DistNew.sum()

@njit('f8[:,:](f8[:,:],f8[:,:],f8[:,:],f8[:],f8[:],f8[:,:],f8,u4[:,:],f8[:,:])')
def DistributionStep2D(Dist,Policy1,Policy2,grid1,grid2,Trans,Mass,Exit,Entry):
    '''
    One Iteration of the Distribution Function given the current policy function. 
    *****************************************************************************
    Policy1,Policy2: np.ndarray((nZ,nX)), policy functions evaluated on state space
    grid1,grid2: np.array(n1),np.array(n2), nonstochastic state grids of lengths n1,n2 such that the state space matrices X1,X2 are
                 computed using combvec(grid1,grid2) 
    Z: np.ndarray((nZ,nX)), stochastic state
    Trans: np.ndarray((nZ,nZ)), transition matrix P such that P[i,j] = Pr(X' = X_j | X = X_i)
    Mass: np.float64, total mass of firms at period t+1
    Exit: np.uint32((nZ,nX)) of 1's at states at which agents exit the economy after end of date t and before start of date t+1
    Entry: np.uint32((nZ,nX)) of *mass* of agents entering the economy after end of date t and before start of date t+1
    '''

    nZ,nL = Policy1.shape
    n2 = grid2.size 

    # Step 1: Interpolation of policy
    iL1,iH1,wL1,iL2,iH2,wL2 = np.zeros(Policy1.shape,dtype="uint32"),np.zeros(Policy1.shape,dtype="uint32"),np.zeros(Policy1.shape,dtype="float64"),np.zeros(Policy1.shape,dtype="uint32"),np.zeros(Policy1.shape,dtype="uint32"),np.zeros(Policy1.shape,dtype="float64")
    for iZ in range(nZ):
        iL1[iZ],iH1[iZ],wL1[iZ] = interpolate_coord_cfunc(grid1,Policy1[iZ])
        iL2[iZ],iH2[iZ],wL2[iZ] = interpolate_coord_cfunc(grid2,Policy2[iZ])
    
    wH1,wH2 = 1-wL1,1-wL2

    # Step 2: Convert Interpolation indices to linear indices
    iLL,wLL = n2*iL1 + iL2, wL1 * wL2
    iLH,wLH = n2*iL1 + iH2, wL1 * wH2 
    iHL,wHL = n2*iH1 + iL2, wH1 * wL2
    iHH,wHH = n2*iH1 + iH2, wH1 * wH2

    # Step 3: Account for Exit
    Dist = Dist * (1-Exit)

    # Step 4: Move Masses
    ind1,ind2 = np.where(Dist/Dist.sum() > 1e-9)
    N = ind1.size
    DistNew = np.zeros(Dist.shape)

    for i in range(N):
        x,y = ind1[i],ind2[i]
        iLLy,iLHy,iHLy,iHHy = iLL[x,y],iLH[x,y],iHL[x,y],iHH[x,y]
        DistNew[x,iLLy] += Dist[x,y] * wLL[x,y] 
        DistNew[x,iLHy] += Dist[x,y] * wLH[x,y]
        DistNew[x,iHLy] += Dist[x,y] * wHL[x,y]
        DistNew[x,iHHy] += Dist[x,y] * wHH[x,y]

    DistNew = np.ascontiguousarray(Trans.T) @ np.ascontiguousarray(DistNew) 

    # Step 5: Entry
    DistNew = DistNew + Entry 
    return DistNew * Mass / DistNew.sum()

@njit 
def DistributionPath1D(Dist0,PolicyPath,X,Trans,Mass,ExitPath,EntryPath):

    T,nZ,nX = PolicyPath.shape 
    DistPath = np.zeros(PolicyPath.shape)
    
    # Assign date 0 distribution
    DistPath[0] = Dist0

    # Iterate
    for t in range(1,T):
        DistPath[t] = DistributionStep1D(DistPath[t-1],PolicyPath[t],X,Trans,Mass,ExitPath[t],EntryPath[t])
    
    return DistPath 

@njit 
def DistributionPath2D(Dist0,Policy1Path,Policy2Path,grid1,grid2,Trans,Mass,ExitPath,EntryPath):

    T,_,_ = Policy1Path.shape
    DistPath = np.zeros(Policy1Path.shape)
    
    # Assign date 0
    DistPath[0] = Dist0 

    # Iterate
    for t in range(1,T):
        DistPath[t] = DistributionStep2D(DistPath[t-1],Policy1Path,Policy2Path,grid1,grid2,Trans,Mass,ExitPath,EntryPath)
    
    return DistPath

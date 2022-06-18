##############################################################################################################
# Routines
# ************************************************************************************************************
#
# Generic routines to solve a variety of problems in dynamic macroeconomics. 
##############################################################################################################

import numpy as np 
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.chebyshev import Chebyshev,cheb2poly
from numba import njit, guvectorize, cfunc
from numpy.random import default_rng
from scipy.special import gamma,gammaincc
from quantecon.markov import tauchen,rouwenhorst

# ------------------------------------------------------------------------------------------------------------
# Extract Data to Numpy Series
# ------------------------------------------------------------------------------------------------------------
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
# Exact matrix determinants and inverses for 3D matrices. X must be an np.ndarray of appropriate dimensions.
# ------------------------------------------------------------------------------------------------------------
def det2(X):
    return X[0,0]*X[1,1] - X[0,1]*X[1,0]

def det3(X):
    m0 = X[1,1]*X[2,2]-X[2,1]*X[1,2]
    m1 = X[1,0]*X[2,2]-X[2,0]*X[1,2]
    m2 = X[1,0]*X[2,1]-X[2,0]*X[1,1]
    return X[0,0]*m0 - X[0,1]*m1 + X[0,2]*m2

def inv3(X):
    det = det3(X) 
    cof = np.zeros((3,3))
    cof[0,0] = det2(X[np.ix_([1,2],[1,2])])
    cof[0,1] = - det2(X[np.ix_([1,2],[0,2])])
    cof[0,2] = det2(X[np.ix_([1,2],[0,1])])
    cof[1,0] = - det2(X[np.ix_([0,2],[1,2])])
    cof[1,1] = det2(X[np.ix_([0,2],[0,2])])
    cof[1,2] = - det2(X[np.ix_([0,2],[0,1])])
    cof[2,0] = det2(X[np.ix_([0,1],[1,2])])
    cof[2,1] = - det2(X[np.ix_([0,1],[0,2])])
    cof[2,2] = det2(X[np.ix_([0,1],[0,1])])
    return cof.T/det

# ------------------------------------------------------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------------------------------------------------------
def combvec_matlab_2D(x1,x2,n1,n2):
    '''
    Construct the output of MATLAB's combvec(x1,x2). 
    '''
    X1 = np.tile(x1,(1,n2))
    X2 = np.kron(x2,np.ones(n1))
    return np.append(X1,X2,0)

def combvec_2D(x1,x2,n1,n2):
    '''
    Construct state space compatible with linear interpolation operators below
    '''
    X1 = np.kron(x1,np.ones(n2))
    X2 = np.tile(x2,(1,n1)).squeeze()
    return np.array([X1,X2])

def combvec_matlab_ND(x_dict,n_vec,n_x):
    '''
    Construct the output of MATLAB's combvec(x1,x2,...,xN). 
    '''
    
    # Preallocate
    N = n_vec.prod()
    X = np.zeros((n_x,N))
    X[0] = np.tile(x_dict[0],int(N/n_vec[0])) 
    X[n_x-1] = np.repeat(x_dict[n_x-1],int(N/n_vec[n_x-1]))

    # Computation
    for i in range(1,n_x-1):
        ntile,nrepeat = n_vec[i+1:].prod(), n_vec[:i].prod()
        X[i] = np.repeat(np.tile(x_dict[i],ntile),nrepeat)
    return X

def combvec_ND(x_dict,n_vec,n_x):
    '''
    Construct the output of MATLAB's combvec(x1,x2,...,xN). 
    '''
    
    # Preallocate
    N = n_vec.prod()
    X = np.zeros((n_x,N))
    X[n_x-1] = np.tile(x_dict[n_x-1],int(N/n_vec[n_x-1])) 
    X[0] = np.repeat(x_dict[0],int(N/n_vec[0]))

    # Computation
    for i in range(1,n_x-1):
        nrepeat,ntile = n_vec[i+1:].prod(), n_vec[:i].prod()
        X[i] = np.repeat(np.tile(x_dict[i],ntile),nrepeat)
    
    return X

def combvecIndex2D(n1,n2,id1,id2):
    '''
    Consider x1 = np.array((n1,)) and x2 = np.array((n2,)) and X = combvec_2D(x1,x2,n1,n2). 
    Consider the 2-D element (a1,a2) where a1 = x1(id1) and a2 = x2(id2)
    This function returns the column index id such that X[:,id] = [a1,a2]
    Useful mainly for bilinear interpolation of the distribution.
    Note: n1 included for discipline.
    '''
    id = n2 * id1 + id2
    return id

def StateSpaces(StatenamesX,StatenamesZ,minvecX,maxvecX,nvecX,rhovecZ,sigvecZ,muvecZ,nvecZ,mvecZ,typecombvecX,typecombvecZ,typevecZ):
    '''
    *****************************************************************************
    Construct a dictionary of state space arrays based on the tensor product grid
    *****************************************************************************
    Assumes that any variables in StatenamesZ are orthogonal AR(1) processes.
    Constructs linear grids.

    * StatenamesX: List-Like of endogenous state variable names.
    * StatenamesZ: List-Like of exogenous state variable names.
    * minvecX,maxvecX,nvecX: List-Like containing min, max and number of elements in each endogenous state variable.
    * rhovecZ,sigvecZ,nvecZ,mvecZ: persistence, shock variance, number of elements and number of sd for approximations to each exog. state variable.
    * typecombvecX: either "matlab" or "python".
    * typevecZ: list with length = # exog variables with each element either "markov" or "rouwenhorst"

    '''

    # Step 0: Preallocate Dictionaries and number of objects
    NX,NZ = np.prod(np.array(nvecX)),np.prod(np.array(nvecZ))
    gridX,StateX,gridZ,StateZ,TransZ,TransTotal = {},{},{},{},{},np.ones((NZ,NZ))
    nstatesX,nstatesZ = len(StatenamesX),len(StatenamesZ)

    # Step 1: Preallocate for each Endogenous and Exogenous variable, Define Grids and Transitions
    StateSize = (NX,NZ)

    for i in range(nstatesX):
        name = StatenamesX[i]
        StateX[name] = np.zeros(StateSize)
        gridX[name] = np.linspace(minvecX[i],maxvecX[i],nvecX[i])
    
    for i in range(nstatesZ):
        name = StatenamesZ[i]
        StateZ[name] = np.zeros(StateSize)
        if typevecZ[i]=='tauchen':
            mc = tauchen(rhovecZ[i],sigvecZ[i],muvecZ[i],mvecZ[i],nvecZ[i])
        elif typevecZ[i] == 'rouwenhorst':
            mc = rouwenhorst(nvecZ[i],muvecZ[i],sigvecZ[i],rhovecZ[i])
        gridZ[name] = mc.state_values
        TransZ[name] = mc.P

    # Step 2: State vectors for endogenous state objects
    if nstatesX == 1:
        # Get name and assign box correctly
        name = StatenamesX[0]
        StateX[name] = np.tile(gridX[name],(NZ,1))

    elif nstatesX == 2:

        # Get names
        name0,name1 = StatenamesX[0],StatenamesX[1]

        # Get grids for combvec operation: faster for 2D case
        grid0,grid1 = gridX[0],gridX[1]
        if typecombvecX == 'python':
            X = combvec_2D(grid0,grid1,nvecX[0],nvecX[1])
        elif typecombvecX == 'matlab':
            X = combvec_matlab_2D(grid0,grid1,nvecX[0],nvecX[1])

        # Construct state spaces that are appropriately shaped, and assign state spaces in order
        StateX[name0] = np.tile(X[0],(NZ,1))
        StateX[name1] = np.tile(X[1],(NZ,1))
    
    else:

        # Use dedicated combvec routine relying on dictionaries of grids
        if typecombvecX == 'python':
            X = combvec_ND(gridX,nvecX,len(nvecX))
        for i in range(len(nvecX)):
            
            # Get names
            name = StatenamesX[i]

            # Construct state spaces that are appropriately shaped, and assign state spaces in order
            StateX[name] = np.tile(X[i],(NZ,1))
    
    # Step 2: State vectors for exogenous state objects
    if nstatesZ == 1:
        # Get name and assign box correctly
        name = StatenamesZ[0]
        StateZ[name] = np.tile(gridZ[name][:,np.newaxis],(1,NX))

    elif nstatesZ == 2:

        # Get names
        name0,name1 = StatenamesZ[0],StatenamesZ[1]

        # Get grids for combvec operation: faster for 2D case
        grid0,grid1 = gridZ[name0],gridZ[name1]
        if typecombvecZ == 'python':
            Z = combvec_2D(grid0,grid1,nvecZ[0],nvecZ[1])
        elif typecombvecZ == 'matlab':
            Z = combvec_matlab_2D(grid0,grid1,nvecZ[0],nvecZ[1])

        # Construct state spaces that are appropriately shaped, and assign state spaces in order
        StateZ[name0] = np.tile(Z[0][:,np.newaxis],(1,NX))
        StateZ[name1] = np.tile(Z[1][:,np.newaxis],(1,NX))
    
    else:

        # Use dedicated combvec routine relying on dictionaries of grids
        if typecombvecZ == 'python':
            Z = combvec_ND(gridZ,nvecZ,len(nvecZ))
        for i in range(len(nvecZ)):
            
            # Get names
            name = StatenamesZ[i]

            # Construct state spaces that are appropriately shaped, and assign state spaces in order
            StateZ[name] = np.tile(Z[i][:,np.newaxis],(1,NX))

    # Step 3: Construct "Total" Transition Matrix P such that
    # P[i,j] = Pr(z_j | z_i) where z_j is the vector of Z-variables at the j'th point.
    if nstatesZ == 1: 
        TransTotal = TransZ[0]

    else:

        # Step 3.1: Map each point in 1:NZ into the index of each variable
        fakeGrid = {} 
        for i in range(nstatesZ):
            fakeGrid[i] = np.arange(nvecZ[i])
        
        if typecombvecZ == 'python':
            indexGrid = combvec_ND(fakeGrid,np.array(nvecZ),nstatesZ)
        elif typecombvecZ == 'matlab':
            indexGrid = combvec_matlab_ND(fakeGrid,np.array(nvecZ),nstatesZ)
        indexGrid = np.uint32(indexGrid)
        
        for i in range(NZ):
            for j in range(NZ):
                for k in range(nstatesZ):
                    name = StatenamesZ[k]
                    idik,idjk = indexGrid[k,i],indexGrid[k,j]
                    TransTotal[i,j] = TransTotal[i,j] * TransZ[name][idik,idjk]

    return gridX,StateX,gridZ,StateZ,TransZ,TransTotal

# ------------------------------------------------------------------------------------------------------------
# Numba-Compatible version of np.tile
# ------------------------------------------------------------------------------------------------------------
@cfunc("float64[:,:](float64[:],uint32,uint32)",nopython=True) 
def TransposedTile(Old,NumOrig,NumRepeat):
    New = np.zeros((NumOrig,NumRepeat))
    for i in range(NumOrig):
        New[i] = np.ones(NumRepeat)*Old[i] 
    return New 

# ------------------------------------------------------------------------------------------------------------
# Interpolation
# -------------
# Note that the routines below work with domains constructed using combvec_XX, not combvec_matlab_XX
# ------------------------------------------------------------------------------------------------------------
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

@cfunc("Tuple((uint32[:],uint32[:],float64[:]))(float64[:], float64[:])",nopython=True)
def interpolate_coord_cfunc(x,xq):
    """
    Adapted from Adrien's notebook (Auclert et al 2021) and compiled as a cfunc to make it 
    numba-importable. 

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

    # preallocate
    xqi = np.zeros(xq.shape,dtype='uint32')
    xqia = np.zeros(xq.shape,dtype='uint32')
    xqpi = np.zeros(xq.shape,dtype='float64')

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
    return xqi,xqia,xqpi

@cfunc('float64[:](float64[:],uint32[:],uint32[:],float64[:])',nopython=True) 
def f_app1D(f, id1, id1h, w1):
    return f[id1] * w1 + f[id1h] * (1 - w1)

@cfunc("float64[:](float64[:],uint32[:],uint32[:],uint32[:],uint32[:],float64[:],float64[:],uint32,uint32)",nopython=True)
def f_app2D(f, id1, id1h, id2, id2h, w1, w2, n1, n2):
    return f[id1 * n2 + id2] * w1 * w2 + f[id1h * n2 + id2] * (1 - w1) * w2 + f[id1 * n2 + id2h] * w1 * (1 - w2) + f[id1h * n2 + id2h] * (1 - w1) * (1 - w2)

@cfunc("float64[:](float64[:],uint32[:],uint32[:],uint32[:],uint32[:],uint32[:],uint32[:],float64[:],float64[:],float64[:],uint32,uint32,uint32)",nopython=True)
def f_app3D(f, id1, id1h, id2, id2h, id3, id3h, w1, w2, w3, n1, n2, n3):
    return (f[id1 * n2 * n3 + id2 * n3 + id3] * w1 * w2 * w3 + 
            f[id1h * n2 * n3 + id2 * n3 + id3] * (1 - w1) * w2 * w3 + 
            f[id1 * n2 * n3 + id2h * n3 + id3] * w1 * (1 - w2) * w3 + 
            f[id1 * n2 * n3 + id2 * n3 + id3h] * w1 * w2 * (1 - w3) + 
            f[id1h * n2 * n3 + id2h * n3 + id3] * (1 - w1) * (1 - w2) * w3 + 
            f[id1h * n2 * n3 + id2 * n3 + id3h] * (1 - w1) * w2 * (1 - w3) + 
            f[id1 * n2 * n3 + id2h * n3 + id3h] * w1 * (1 - w2) * (1 - w3) + 
            f[id1h * n2 * n3 + id2h * n3 + id3h] * (1 - w1) * (1 - w2) * (1 - w3))

@cfunc("float64[:](float64[:],float64[:],float64[:])",nopython=True)
def approx_1D(f, x1, grid1):
    id1, id1h, w1 = interpolate_coord_cfunc(grid1, x1)
    return f_app1D(f, id1, id1h, w1)

@cfunc("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],uint32,uint32)",nopython=True)
def approx_2D(f, x1, x2, grid1, grid2, n1, n2):
    id1,id1h,w1 = interpolate_coord_cfunc(grid1, x1)
    id2,id2h,w2 = interpolate_coord_cfunc(grid2, x2)
    res = f_app2D(f, id1, id1h, id2, id2h, w1, w2, n1, n2)
    return res

@cfunc("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],uint32,uint32,uint32)",nopython=True)
def approx_3D(f, x1, x2, x3, grid1, grid2, grid3, n1, n2, n3):
    id1, id1h, w1 = interpolate_coord_cfunc(grid1, x1)
    id2, id2h, w2 = interpolate_coord_cfunc(grid2, x2)
    id3, id3h, w3 = interpolate_coord_cfunc(grid3, x3)
    return f_app3D(f, id1, id1h, id2, id2h, id3, id3h, w1, w2, w3, n1, n2, n3)

@cfunc("float64[:,:](float64[:,:],float64[:,:],float64[:,:],uint32,uint32)",nopython=True)
def DenserPolicyFunction(Policy,StateSpace,StateSpaceD,nZ,nX):
    '''
    Expands density in one dimension
    ********************************
    Policy: (nZ,nX)
    StateSpace: (nZ,nX)
    StateSpaceD: (nZ,nX*dfac)
    DenserPolicy: (nZ,nX*dfac)
    '''
    DenserPolicy = np.zeros(StateSpaceD.shape)
    for iZ in range(nZ):
        DenserPolicy[iZ] = approx_1D(Policy[iZ],StateSpaceD[iZ],StateSpace[iZ])

    return DenserPolicy

@cfunc("float64[:,:](float64[:,:],float64[:,:],float64[:,:],float64[:],float64[:],uint32,uint32,uint32)",nopython=True)
def DenserPolicyFunction2D(Policy,StateSpace1D,StateSpace2D,grid1,grid2,nZ,n1,n2):
    '''
    Expands density in two dimensions
    *********************************
    '''
    DenserPolicy = np.zeros(StateSpace1D.shape)
    for iZ in range(nZ):
        DenserPolicy[iZ] = approx_2D(Policy[iZ],StateSpace1D[iZ],StateSpace2D[iZ],grid1,grid2,n1,n2)
    
    return DenserPolicy

@cfunc("float64(float64[:,:],float64[:,:],float64[:,:],float64[:,:],uint32,uint32)",nopython=True)
def DenseAggregate(Policy,Dist,StateSpace,StateSpaceD,nZ,nX):
    PolicyD = DenserPolicyFunction(Policy,StateSpace,StateSpaceD,nZ,nX)
    Aggregate = (PolicyD * Dist).sum()
    return Aggregate

@cfunc("float64(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:],float64[:],uint32,uint32,uint32)",nopython=True)
def DenseAggregate2D(Policy,Dist,StateSpace1D,StateSpace2D,grid1,grid2,nZ,n1,n2):
    PolicyD = DenserPolicyFunction2D(Policy,StateSpace1D,StateSpace2D,grid1,grid2,nZ,n1,n2)
    Aggregate = (PolicyD * Dist).sum()
    return Aggregate

# ------------------------------------------------------------------------------------------------------------
# Solvers
# ------------------------------------------------------------------------------------------------------------
def forward_derivative(f,x,h=1e-8):
    if len(x) > 1:
        h = h*np.ones(len(x))
    return ( f(x+h) - f(x) )/h

def backward_derivative(f,x,h=1e-8):
    if len(x) > 1:
        h = h*np.ones(len(x))
    return ( f(x) - f(x-h) )/h

def central_derivative(f,x,h=1e-8):
    if len(x) > 1: 
        h = h*np.ones(len(x))
    return ( f(x+h) - f(x-h) )/(2*h)

def Jacobian_2d(f,x,h=1e-8):
    '''
    f accepts a 2-d vector as input and outputs a 2-d vector.
    '''
    # Preallocate Jacobian
    Jac = np.zeros((2,2))

    # Compute 4 corners
    f00 = f(x+np.array([h,0]))[0]
    f01 = f(x+np.array([0,h]))[0]
    f10 = f(x+np.array([h,0]))[1]
    f11 = f(x+np.array([0,h]))[1]

    # Assign Derivatives
    Jac[0,0] = (f00 - f(x))/h
    Jac[0,1] = (f01 - f(x))/h
    Jac[1,0] = (f10 - f(x))/h
    Jac[1,1] = (f11 - f(x))/h
    return Jac

def Jacobian(f,x,y,h=1e-5):
    '''
    Calculate Numerical Jacobian of a function f at 
    x. Define y = f(x).

    Recall that the Jacobian is defined as 
    J[i,j] = df_i/dx_j

    But because Python is column-major, it's a bit 
    faster to compute the transpose and return the transposed matrix.
    '''
    
    nx,ny = x.shape[0],y.shape[0]
    Jac = np.zeros((nx,ny))

    for i in range(nx):
        
        # Raise i'th component of x
        dx = np.zeros(nx)
        dx[i] = h 

        # Construct Jacobian
        Jac[i,:] = (f(x+dx) - y)/h

    return Jac.T

def solver_1d_bisection(f,x0,x1,h=1e-9,maxiter=1000,tol=1e-8,relax=1,noisily=1):
    '''
    Similar to Matlab's FZERO, finds a zero between x0 and x1. 
    f must be a routine that accepts a single np.float64 input.
    NOTE: Does not find *all* zeros, just the first one that a bisection algorithm would detect. 
    Automatically jumps basins if it thinks it is getting stuck. 
    I think this is Guaranteed to find *a* zero, if one exists. 
    '''
    if f(x0) * f(x1) > 0:
        print(f'error: f(x0),f(x1) must have different signs, but we have f(x0)={f(x0)} and f(x1)={f(x1)}')
        return None
    else:
        it,err = 0,100
        x00,x10 = x0,x1
        while (abs(err)>tol) & (it<=maxiter):
            it+=1
            x = (x0+x1)/2
            err = f(x)
            fl,fh = (f(x)-f(x-h))/h,(f(x+h)-f(x))/h
            if ((fl * fh <= 0) | (abs(fl)<tol*1e-2) | (abs(fh)<tol*1e-2)) & (abs(err)>tol):
                '''
                This means the solver is stuck at a local min/max, but this isn't necessarily a solution. 
                '''
                flimL,flimH = f(x00),f(x10)
                if flimL * err < 0: 
                    x0 = x00
                    x1 = x1*(1-relax) + relax*x
                elif flimH * err < 0:
                    x1 = x10
                    x0 = x0*(1-relax) + relax*x   
                if noisily == 1:
                    print(f'I think I may be stuck at a critical point - I will be jumping basins to [{x0},{x1}] on the next iteration.')                  
            else: 
                if ((err > 0) & (fl > 0) & (fh > 0)) | ((err < 0) & (fl < 0) & (fh < 0)):
                    x1 = x1*(1-relax) + relax*x  
                elif ((err < 0) & (fl > 0) & (fh > 0)) | ((err > 0) & (fl < 0) & (fh < 0)):
                    x0 = x0*(1-relax) + relax*x
            if noisily > 0:
                if it%noisily == 0:
                    print(f'iter {it}/{maxiter}, error {np.abs(err)}, interval = [{x0},{x1}], fl = {fl}, fh = {fh}')
        return x,err

def solver_1d_bisection_intervalled(f,x0,x1,Npoints = 100,h=1e-9,maxiter=1000,tol=1e-8,relax=1,noisily=1):
    '''
    Similar to Matlab's FZERO, finds zeros between x0 and x1. 
    f must be a routine that accepts a single np.float64 input.
    NOTE: Does not find *all* zeros, just the first ones that a bisection algorithm would detect. 
    Automatically jumps basins if it thinks it is getting stuck. 
    I think this is Guaranteed to find *a* zero, if one exists. 
    '''
    # Step 0: Split Interval
    xrange = np.linspace(x0,x1,Npoints)
    
    # Step 1: Find all intervals in which a zero is likely to exist at this resolution
    startpoints = []
    for i in range(len(xrange)-1):
        x0,x1 = xrange[i],xrange[i+1]
        if f(x0)*f(x1)<0:
            if noisily > 0:
                print(f'probably a zero in [{x0},{x1}]: function values are [{f(x0)},{f(x1)}]')
            startpoints.append(i)
        else:
            if noisily > 0:
                print(f'probably no zero in [{x0},{x1}]: function values are [{f(x0)},{f(x1)}]')
    startpoints = [int(i) for i in startpoints]
    endpoints = [int(i+1) for i in startpoints]
    
    # Step 2: Find a zero in each interval
    solutions = np.zeros(len(startpoints))
    for j in range(len(startpoints)):
        x0,x1 = xrange[startpoints[j]],xrange[endpoints[j]]
        solutions[j],_ = solver_1d_bisection(f,x0,x1,h=h,maxiter=maxiter,tol=tol,relax=relax,noisily=noisily)
    
    # Return results
    return xrange[startpoints],xrange[endpoints],solutions

def minimize_1d_newton(f,x0,maxiter=10000,tol=1e-8,seed=1,noisily=0):
    '''
    Find minimum of a 1-dimensional function by Nesterov's accelerated gradient descent method.
    Note that f must be a smooth convex function handle. 
    '''
    x,y = np.array([x0]),np.array([x0])
    
    lambda_seq = np.zeros(maxiter+1)
    for i in range(1,maxiter+1):
        lambda_seq[i] = 0.5 + 0.5*np.sqrt(1+(4*(lambda_seq[i-1]**2)))
    
    gamma_seq = (1-lambda_seq[:maxiter])/lambda_seq[1:]
    gamma_seq = gamma_seq[1:]

    # Calculate Lipschitz coefficient by random sampling
    NLips = 100
    rng = default_rng(seed)
    xpoints = x0*(1 + 10*rng.random(NLips))
    
    grads = np.array([central_derivative(f,np.array([x])) for x in xpoints])
    beta = np.max(np.abs(grads))
    print(beta)

    iter,error = 0,1

    while (error>=tol) & (iter<=maxiter):
        
        # Old positions
        x_0,y_0 = x,y
        print(x_0)
        print(y_0)

        # Calculate gradient
        grad = central_derivative(f,x_0)

        # Updates with Nesterov Momentum correction
        y_1 = x_0 - grad/beta
        x_1 = (1-gamma_seq[iter])*y_1 + gamma_seq[iter]*y_0
        
        # Error 
        error = abs(central_derivative(f,x_1))

        # Update 
        x,y = x_1,y_1
        iter += 1
        
        if noisily != 0:
            if iter % noisily == 0:
                print(f'iter {iter}, fval = {f(x)}, error = {error}')

    xmin,fmin = x,f(x)

    return xmin,fmin

def minimize_1d_unimodal(f,x0,x1,maxiter=10000,tol=1e-8,noisily=0):
    '''
    Minimize a unimodal function f on an interval [x0,x1]
    '''
    # Fix types
    x0,x1 = np.array([x0]),np.array([x1])
    
    # Endpoint derivatives
    DL,DH = forward_derivative(f,x0),backward_derivative(f,x1)
    #DL,DH = DL.squeeze(),DH.squeeze()
    if (DL>0) & (DH>0):
        # Minimium at the left endpoint
        xmin,fmin = x0,f(x0)
    elif (DL<0) & (DH<0):
        # Maximum at the right endpoint
        xmin,fmin = x1,f(x1)
    elif (DL>0) & (DH<0):
        # Some mistake
        print(f'function increasing at left end and decreasing at right end: something is wrong.')
        xmin,fmin = np.NaN,np.NaN
    else:
        # Interior optimum! Use iterative algorithm.
        iter,error = 0,1
        while (error>tol)&(iter<=maxiter):
            x = 0.5*(x0+x1)
            Dx = central_derivative(f,x)
            error = abs(Dx)
            iter += 1
            if Dx > 0:
                x1 = x
            else:
                x0 = x
            if noisily!=0:
                if iter%noisily == 0:
                    print(f'iter {iter}, interval [{x0},{x1}]')  
        xmin,fmin = x,f(x)
    return xmin,fmin

def SimpleNewtonSolver(f,xinit,JacRoutine=None,yinit=None,tol=1e-5,maxit=1000,backtrack_c=0.5):
    '''
    Line search solver for root of a multidimensional function.
    ***********************************************************
    Essentially from the routine in Auclert et al (2021). 
    Uses backtracking to avoid basin problems (somewhat). 
    Main issue: relies on potentially expensive Jacobian evaluations. 
    
    Note: if you have a closed-form function handle that can evaluate the Jacobian 
    and is a function of two arguments, the location x and the function value at the
    point, then set that in JacRoutine. Else, the solver uses a numerical approximation.
    '''

    # Initialize
    x,y = xinit,yinit 
    if y is None:
        y = f(x)
    
    # Define Jacobian Routine
    if JacRoutine is None: 
        JacRoutine = lambda x,y: Jacobian(f,x,y,tol*0.1)
    
    # Begin Iterations
    it = 0
    while (y>tol) & (it<=maxit):

        # Check if converged
        if np.max(np.abs(y))<=tol:            
            return x,y

        J = JacRoutine(x,y)
        dx = np.linalg.solve(J,-y)
        for bcount in range(30):
            try:
                y_next = f(x+dx)
            except ValueError:
                dx *= backtrack_c
            else:
                predicted_improvement = -np.sum((J @ dx) * y) * ((1 - 1 / 2 ** bcount) + 1) / 2
                actual_improvement = (np.sum(y ** 2) - np.sum(y_next ** 2)) / 2     
                if actual_improvement < predicted_improvement / 2:
                    dx *= backtrack_c
                else: 
                    y = y_next
                    x += dx 
                    break 

        else:
            raise ValueError('Too many backtracks (bad initial guess)')
    
    if it == maxit:
        raise ValueError(f'no convergence after {maxit} iterations.')
    
def broyden_update(J,dx,dy):
    """Returns Broyden update to approximate Jacobian J, given that last change in inputs to function
    was dx and led to output change of dy."""
    return J + np.outer(((dy - J @ dx) / np.linalg.norm(dx) ** 2), dx)

def SimpleNewtonSolver(f,xinit,JacRoutine=None,yinit=None,tol=1e-5,maxit=1000,backtrack_c=0.5):
    '''
    Line search solver for root of a multidimensional function with Broyden updating for the Jacobian
    *************************************************************************************************
    Essentially from the routine in Auclert et al (2021). 
    Uses backtracking to avoid basin problems (somewhat). 
    Main issue: relies on potentially expensive Jacobian evaluations. 
    
    Note: if you have a closed-form function handle that can evaluate the Jacobian 
    and is a function of two arguments, the location x and the function value at the
    point, then set that in JacRoutine. Else, the solver uses a numerical approximation.
    '''

    # Initialize
    x,y = xinit,yinit 
    if y is None:
        y = f(x)
    
    # Define Jacobian Routine
    if JacRoutine is None: 
        JacRoutine = lambda x,y: Jacobian(f,x,y,tol*0.1)
    
    # Begin Iterations
    it = 0
    while (y>tol) & (it<=maxit):

        # Check if converged
        if np.max(np.abs(y))<=tol:            
            return x,y

        if it == 0:
            J = JacRoutine(x,y)

        if len(x) == len(y):
            dx = np.linalg.solve(J, -y)
        elif len(x) < len(y):
            dx = np.linalg.lstsq(J, -y, rcond=None)[0]

        # Backtracking 
        for bcount in range(30):
            try:
                y_next = f(x+dx)
            except ValueError:
                dx *= backtrack_c
            else:
                J = broyden_update(J,dx,y_next-y)   
                y = y_next
                x += dx 
                break 

        else:
            raise ValueError('Too many backtracks (bad initial guess)')
    
    if it == maxit:
        raise ValueError(f'no convergence after {maxit} iterations.')

# ------------------------------------------------------------------------------------------------------------
# Genetic Routines for minimization and root finding
# ------------------------------------------------------------------------------------------------------------
def genetic_iterate(func,population,LimitsL,LimitsH,immigration_rate,death_rate,pr_crossover,pr_mutation,mutation_factor,Npop,Nargs,blx_alpha=0.5,seed=2):
    '''
    **************************
    One Genetic Algorithm Step
    **************************
    Note that all routines here treat lower function values as higher in fitness scale, and
    that lower values for fitness are better. 

    death_rate: kill off this fraction of the population's least fit agents
    pr_crossover: probability that any given agent gets to participate in reproduction
    pr_mutation: probability that any given agent gets mutated

    When mutation occurs, each element gets multiplied by either 
    1+mutation_factor 
    or 
    1-mutation_factor
    (see Heer and Maussner or minimization routine for the operator
    used to compute the mutation factor.)

    This routine implements a version of Heer and Maussner (2009)'s recommendations for 
    different modes of reproduction. Every reproducing couple has 8 children based on 
    different approaches to crossover. The fittest are retained at the end. 

    The fittest agent in each generation is guaranteed to be at least as fit as the input
    generation to ensure that the algorithm has a monotonicity built into it. 

    This algorithm also allows for "immigration" both before and after reproduction occurs. 
    Immigration adds agents at random and ensures that there is always some genetic diversity.
    Half of the immigrants have the chance to mate with natives, and the other half are added
    post reproduction.
    '''
    
    rng = default_rng(seed)

    # Step 1: Immigration
    Nimmig = int( Npop*immigration_rate/2 )
    immigrants = np.tile(LimitsL,(Nimmig,1)) + rng.random((Nimmig,Nargs)) * np.tile(LimitsH-LimitsL,(Nimmig,1))
    population = np.append(population,immigrants,0)

    # Step 2: Compute Fitness
    Npop_current = Npop + Nimmig
    fitness = np.array([func(population[i]) for i in range(Npop_current)])
    population = population[np.argsort(fitness),:]
    fitness = fitness[np.argsort(fitness)]

    # Step 3: Deaths (kill least fit agents)
    Nsurvivors = int( (1-death_rate)*Npop )
    population = population[:Nsurvivors,:]
    fitness = fitness[:Nsurvivors]
    Npop_current = Nsurvivors

    # Step 4: Reproduction and Crossover
    cutpoints,crossover_rands = rng.integers(low=0,high=Nargs-1,size=int(Npop_current)),rng.random(int(Npop_current))
    crossprobs = pr_crossover * (fitness - fitness.min()) / (fitness.max()-fitness.min())
    crossover_population = population[crossover_rands<=crossprobs,:]

    for j in range(len(crossover_population[:,0])-1):
        
        mom,dad = crossover_population[j],crossover_population[j+1]
        kid = np.zeros((8,Nargs))

        # Kids, part 1: Simple Crossover
        kid[0] = np.append(mom[:int(cutpoints[j])],dad[int(cutpoints[j]):])
        kid[1] = np.append(mom[:int(cutpoints[j])],dad[int(cutpoints[j]):])

        # Kids, part 2: Shuffle Crossover
        rands = rng.random(Nargs)
        kid[2] = np.array([mom[i]*(rands[i]<=0.5) + dad[i]*(rands[i]>0.5) for i in range(Nargs)])
        kid[3] = np.array([dad[i]*(rands[i]<=0.5) + mom[i]*(rands[i]>0.5) for i in range(Nargs)])
        
        # Kids, part 3: Arithmetic Crossover
        rand = rng.random()
        kid[4] = rand*mom + (1-rand)*dad
        kid[5] = rand*dad + (1-rand)*mom 

        # Kids, part 4: BLX-Alpha
        pmax = np.maximum(mom,dad)
        pmin = np.minimum(mom,dad)
        Delta = pmax-pmin
        rand = rng.random(2)
        low,hi = pmin-blx_alpha*Delta,pmax+blx_alpha*Delta
        kid[6] = low + rand[0]*(hi-low)
        kid[7] = low + rand[1]*(hi-low)

        # Add family to population
        population = np.append(population,kid,0)

    # Step 5: Mutation
    Npop_current = len(population[:,0])
    mutation_rands = rng.random(Npop_current)
    ranges = np.array([population[:,i].max()-population[:,i].min() for i in range(Nargs)])

    for j in np.where(mutation_rands<=pr_mutation):
        rand = rng.random(Nargs)
        mutant = population[j] * (1+ (ranges * mutation_factor * ((-1)**(rand<=0.5))))
        population = np.append(population,mutant,0) 

    # End-period Immigration
    Nimmig = int( Npop*immigration_rate/2 )
    immigrants = np.tile(LimitsL,(Nimmig,1)) + rng.random((Nimmig,Nargs)) * np.tile(LimitsH-LimitsL,(Nimmig,1)) 
    population = np.append(population,immigrants,0)

    # Ensure Limits are Respected
    Npop = len(population[:,0])
    L,H = np.tile(LimitsL,(Npop,1)),np.tile(LimitsH,(Npop,1))
    population = np.minimum(np.maximum(population,L),H)

    # Final Selection Step
    fitness = np.array([func(population[i]) for i in range(len(population[:,0]))])
    population = population[np.argsort(fitness),:]
    fitness = fitness[np.argsort(fitness)]
    population,fitness = population[:Npop,:],fitness[:Npop]

    return population,fitness

def genetic_minimize(   func,Npop,Nargs,LimitsL,LimitsH,
                        immigration_rate_init=0.1,death_rate=0.1,pr_crossover=0.7,
                        blx_alpha=0.5,maxit=100,tol = 1e-9,
                        mutation_b=5,mutation_pi1 =0.15,mutation_pi2=0.33,
                        noisily=10,seed=2,keep_history=False):
    '''
    ************************************
    Minimization via a Genetic Algorithm
    ************************************
    func : callable that takes a single input of size (Nargs,) and outputs an np.float64()
    Npop : number of agents in population. 
    Nargs : dimension of each vector
    LimitsL,LimitsH : np.array(Nargs) of lower/upper limits

    If keep_history is set to True, then the population object has an extra dimension. 
    Note that this could mess up memory. 
    '''
    # Step 1: Initialize Population
    rng = default_rng(seed)
    population_init = np.tile(LimitsL,(Npop,1)) + rng.random((Npop,Nargs)) * np.tile(LimitsH-LimitsL,(Npop,1)) 

    # Step 2: Construct Mutation Factors and Probabilities
    mutation_factor = 1 - (rng.random(maxit) ** (1 - ((np.arange(maxit)/(maxit-1))**mutation_b)))
    pr_mutation = mutation_pi1 + mutation_pi2/np.arange(maxit)
    immigration_rate = immigration_rate_init * np.exp(-np.arange(maxit)*10/maxit)

    # Step 3: Begin Iterations
    err,it = 1,0 
    if keep_history == True: 
        population,fitness = np.zeros((maxit+1,Npop,Nargs)),np.zeros((maxit+1,Npop))
        population[0],fitness[0] = population_init,[func(population_init[i]) for i in range(Npop)]

        while (err>tol) & (it<=maxit-1):
            
            population[it+1],fitness[it+1] = genetic_iterate(func,population[it],LimitsL,LimitsH,immigration_rate[it],death_rate,pr_crossover,pr_mutation[it],mutation_factor[it],Npop,Nargs,blx_alpha,seed)
            
            # Manage Genetic Diversity through immigration
            genetic_diversity = (population[it+1].std(0)/population[it+1].mean(0)).min()
            if genetic_diversity < 1e-5:
                immigration_rate[it:] = np.maximum(0.4,np.minimum(1,immigration_rate[it:]*1.5))
            
            if noisily > 0:
                if it % noisily == 0:
                    print(f'iteration {it}, current best fit {fitness[0]}, fittest population member {population[it+1,0]}, genetic diversity by comp {population[it+1].std(0)/population[it+1].mean(0)}')
            
            it+=1
            err = np.abs(fitness[0])

    
    else: 
    
        population = population_init.copy()
        while (err>tol) & (it<=maxit-1):

            population_init = population.copy()
            population,fitness = genetic_iterate(func,population_init,LimitsL,LimitsH,immigration_rate[it],death_rate,pr_crossover,pr_mutation[it],mutation_factor[it],Npop,Nargs,blx_alpha,seed)

            # Manage Genetic Diversity through immigration
            genetic_diversity = (population.std(0)/population.mean(0)).min()
            if genetic_diversity < 1e-5:
                immigration_rate[it:] = np.maximum(0.4,np.minimum(1,immigration_rate[it:]*1.5))
            
            if noisily > 0:
                if it % noisily == 0:
                    print(f'iteration {it}, current best fit {fitness[0]}, fittest population member {population[0]}, genetic diversity by comp {population.std(0)/population.mean(0)}')

            it+=1 
            err = np.abs(fitness[0])

    return population,fitness

def genetic_root(   func0,Npop,Nargs,LimitsL,LimitsH,
                    immigration_rate_init=0.1,death_rate=0.1,pr_crossover=0.7,
                    blx_alpha=0.5,maxit=100,tol = 1e-3,
                    mutation_b=5,mutation_pi1 =0.15,mutation_pi2=0.33,
                    noisily=10,seed=2):

    '''
    Root-finding via a Genetic Algorithm. Similar to the minimization routine.
    '''
    func = lambda x: func0(x)**2 

    # Step 1: Initialize Population
    rng = default_rng(seed)
    population = np.tile(LimitsL,(Npop,1)) + rng.random((Npop,Nargs)) * np.tile(LimitsH-LimitsL,(Npop,1)) 

    # Step 2: Construct Mutation Factors and Probabilities
    mutation_factor = 1 - (rng.random(maxit) ** (1 - ((np.arange(maxit)/(maxit-1))**mutation_b)))
    pr_mutation = mutation_pi1 + mutation_pi2/np.maximum(1,np.arange(maxit))
    immigration_rate = immigration_rate_init * np.exp(-np.arange(maxit)*10/maxit)

    # Step 3: Begin Iterations
    it,err = 0,1
    while (err>=tol) & (it<maxit):
        
        # Iteration
        population_init = population.copy()
        population,fitness = genetic_iterate(func,population_init,LimitsL,LimitsH,immigration_rate[it],death_rate,pr_crossover,pr_mutation[it],mutation_factor[it],Npop,Nargs,blx_alpha,seed)

        # Manage Genetic Diversity through immigration
        genetic_diversity = (population.std(0)/population.mean(0)).min()
        if genetic_diversity < 1e-5:
            immigration_rate[it:] = np.maximum(0.4,np.minimum(1,immigration_rate[it:]*1.5))

        # Error Check
        it+=1
        err = fitness[0]

        if noisily > 0:
            if it % noisily == 0:
                print(f'iteration {it}, current best fit {fitness[0]}, fittest population member {population[0]}, genetic diversity by comp {population.std(0)/population.mean(0)}')

    return population,fitness

# ------------------------------------------------------------------------------------------------------------
# Chebyshev Polynomial Routines: First Kind
# *****************************************
# NOTE: Mimics routines written for grids constructed using combvec_matlab. 
# Also, note that when you ask for approximations with n_vec = n, the routines return all polynomials 
# T_0, ..., T_n , which is n+1 polynomials. 
# ------------------------------------------------------------------------------------------------------------
def DomainChange_Original_to_Cheby(DomainOriginal,limL,limH):
    '''
    Transform X defined on a grid from limL to limH to a Chebyshev Domain [-1,1]**N
    DomainOriginal: np.ndarray((nDim,nX))
    '''
    nDim = len(limL)
    if DomainOriginal.shape[0] != nDim:
        print(f'Shape Mismatch: Original domain has shape {DomainOriginal.shape}, but must have first dimension {nDim}')
        raise ValueError
    else:
        DomainCheby = np.zeros(DomainOriginal.shape)
        for i in range(nDim):
            
            # Respect Limits
            DomainOriginal[i] = np.maximum(np.minimum(DomainOriginal[i],limH[i]),limL[i])
            
            # Translation
            DomainCheby[i] = -1 + 0.5*(DomainOriginal[i]-limL[i])/(limH[i]-limL[i])            

        return DomainCheby

def DomainChange_Cheby_to_Original(DomainCheby,limL,limH):
    '''
    Transform X defined on a Chebyshev Domain [-1,1]**N to a grid from limL to limH
    DomainCheby: np.ndarray((nDim,nX))
    '''
    nDim = len(limL)
    if DomainCheby.shape[0] != nDim:
        print(f'Shape Mismatch: Original domain has shape {DomainCheby.shape}, but must have first dimension {nDim}')
        raise ValueError
    else:
        DomainOriginal = np.zeros(DomainCheby.shape)

        # Translation
        DomainCheby = np.maximum(np.minimum(DomainCheby,1),-1)
        for i in range(nDim):

            # Respect Limits
            DomainOriginal[i] = limL[i] + (limH[i]-limL[i])*(DomainCheby[i] + 1)/2

        return DomainOriginal

def cheby_coeff(n,m):
    coeffcheb = np.zeros(n)
    coeffcheb[n-1] = 1
    cheb = Chebyshev(coeffcheb)
    coef = cheb2poly(cheb.coef)
    return coef[m]

def cheby_grid(n):
    '''
    Construct Chebyshev Polynomials T_0,T_1, ..., T_n evaluated at nodes of T_(n+1)
    Note: box[:,i] is the value of T_i evaluated at each node of T_n+1
    outputs are n+1-dimensional. 
    '''
    grid = -np.cos((2*np.arange(n+1)-1)*np.pi/(2*n))
    box = np.ones((n+1,n+1))
    box[:,1] = grid
    for j in range(2,n+1):
        box[:,j] = 2*grid*box[:,j-1] - box[:,j-2]
    return grid,box

def cheby_funcs(nDim,X):
    '''
    Compute Chebyshev Polynomials at points X. 
    X: np.ndarray(nPoints), X[i] lies in [-1,1] for all i. 
    nDim: int >= 1
    output: box is ((nPoints,nDim+1))
    '''
    nPoints = len(X)
    box = np.ones((nPoints,nDim+1))
    box[:,1] = X
    for i in range(2,nDim+1):
        box[:,i] = 2*X*box[:,i-1] - box[:,i-2]
    return box

def cheby_Tensorize(X,n_vec):
    '''
    Construct Tensor characterizing values of a Chebyshev polynomial approximation at each point in X
    X: np.ndarray((nDim,nPoints))
    n_vec: np.ndarray(nDim), n_vec[i] is the dimension-i order of approximation.
    '''

    # Dimensions
    nDim,nPoints = X.shape

    # Preallocate
    T = np.ones(((n_vec+1).prod(),nPoints))

    # Construct a Domain for Chebyshev Polynomials
    Orders = np.array([np.arange(n_vec[0]+1)])
    for i in range(1,nDim):
        Orders = combvec_matlab_2D(Orders,np.array([np.arange(n_vec[i]+1)]),Orders[0].shape,n_vec[i]+1)
    
    # Calculate Chebyshev Polynomials
    Chebys = {}
    for i in np.arange(nDim):
        Chebys[i] = cheby_funcs(n_vec[i],X[i])

    # Compute Product Polynomials
    for i in range(n_vec.prod()):
        for ii in range(nDim):
            T[i,:] = T[i,:] * Chebys[ii][:,int(Orders[ii,i])]

    return T

def cheby_eval(X,gamma,n_vec,limL,limH):
    '''
    Compute linear combination gamma of Chebyshev polynomials based on product polynomials
    note: X is np.ndarray((nDim,nPoints))
    output is np.ndarray((nPoints,1))
    '''
    X = DomainChange_Original_to_Cheby(X,limL,limH)
    return cheby_Tensorize(X,n_vec).T @ gamma

def cheby_project(f,limL,limH,n_vec,X):

    '''
    Projection of function handle f onto space of cheby polynomials. 
    Based on LeastSquares residual minimization at points in X. 
    
    X: np.ndarray((nDim,nPoints))
    limL,limH: np.ndarray(nDim)
    n_vec: np.ndarray(nDim)
    f: function handle that takes one vector argument as an input.
    '''

    fvals = np.array([f(X[:,i]) for i in range(len(X[0]))])
    domain = DomainChange_Original_to_Cheby(X,limL,limH)
    
    # Compute function values
    T = cheby_Tensorize(domain,n_vec)
    gamma = np.linalg.lstsq(T.T,fvals,rcond=None)[0]

    return gamma

# ------------------------------------------------------------------------------------------------------------
# Simple Integrals of Polynomial Functions of LogNormally Distributed Variables with mean mu, variance sigma2
# ------------------------------------------------------------------------------------------------------------
def LogNormalMoment(a,mu=0,sigma=1,Nquad=11):
    '''
    Compute E(Z**a) where logZ is normal by Gauss-Hermite Quadrature
    '''
    nodes,weights = hermgauss(Nquad)
    nodes = np.sqrt(2)*sigma*nodes + mu
    f = np.exp(nodes*a)
    mom = (f*weights).sum()/np.sqrt(np.pi)
    return mom 

def GaussHermiteQuadrature(f,mu=0,sigma=1,Nquad=11):
    '''
    Compute the Gauss-Hermite Quadrature Integral of f. 
    f: function handle accepting one input which is normally distributed with mean mu and std sigma. 
    '''
    nodes,weights = hermgauss(Nquad)
    xvalues = np.sqrt(2)*sigma*nodes + mu 
    fvec = np.array([f(x) for x in xvalues])
    return (fvec*weights).sum()/np.sqrt(np.pi)

def PreComputationLogNormalIntegrals(sigE,order,basefunc='polynomial'):
    '''
    Let F be a function of X and Z=exp(z), where

    z' = rhoZ * z + e, 

    where e~N(0,sigE^2). Suppose that we are trying to approximate F by a linear combination 
    of polynomials in X and exp(z), i.e 

    Fhat = np.dot(b , Poly(X,z) )
    
    In economic models, one might eg approximate the value function V(k,z) in a NGM as a 
    linear combination of ordinary polynomials in k,z; to second order, 
    
    V(k,z) = b0 + b1*k + b2*exp(z) + b3*(k**2) + b4*(exp(z)**2) + b5*k*exp(z)
    
    In this case, note that 
    
    E(Fhat(X',Z')|Z) = np.dot( np.dot(b(x,exp(rhoZ*z)), E(exp(e)) )

    due to the linearity of the expectations operator. Precomputation recognizes that E(exp(e)) can be
    solved for *analytically* if the e's are normally distributed, or numerically if the z's are not 
    normal. 

    '''

    if basefunc == 'polynomial':
        '''
        PreCompCoeffs[i] is the precomputed expectation for any terms in which z is raised to a power i.
        '''
        PreCompCoeffs = np.array([np.exp( 0.5 * ((i * sigE)**2) ) for i in np.arange(order+1)])

    if basefunc == 'chebyshev':
        '''
        PreCompCoeffs[i,j] is the precomputed expectation for any terms in a Tensor grid containing the 
        jth Chebyshev Polynomial in z in which z is raised to the power i. 
        '''
        PreCompCoeffs_power = np.array([np.exp( 0.5 * ((i * sigE)**2) ) for i in np.arange(order+1)])
        PreCompCoeffs_cheby = np.array([cheby_coeff(order+1,i) for i in np.arange(order+1)])
        PreCompCoeffs = np.zeros((order+1,order+1))
        for i in range(order+1):
            for j in range(order+1):
                PreCompCoeffs[i,j] = PreCompCoeffs_cheby[j] * PreCompCoeffs_power[i]

    return PreCompCoeffs

# -----------------------------------------------------------------------------------------------------------
# Pareto Distribution Moments (JIT-compatible)
# -----------------------------------------------------------------------------------------------------------
@njit 
def Pareto_survival(X,Xmin,TailIndex):
    '''
    Survival Function of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        surv = 1
    else:
        surv = (Xmin/X)**TailIndex
    return surv

@njit 
def Pareto_cdf(X,Xmin,TailIndex):
    '''
    CDF of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        cdf = 0
    else:
        cdf = 1 - ((Xmin/X)**TailIndex)
    return cdf

@njit 
def Pareto_pdf(X,Xmin,TailIndex):
    '''
    PDF of a Pareto Distribution with (Location,Shape) = (Xmin,TailIndex)
    '''
    if X < Xmin:
        pdf = 0
    else:
        pdf = TailIndex * (Xmin**TailIndex) * (X**(-(TailIndex+1)))
    return pdf

@njit 
def Pareto_CondExp_Int_Above(X,Xmin,TailIndex):
    '''
    Calculate int_X^infty Z f(Z) dZ where f(Z) is the Pareto distribution with (Location,Shape) = (Xmin,TailIndex)
    IMPORTANT: need TailIndex > 1
    '''
    if X < Xmin: 
        I = (TailIndex/(TailIndex-1)) * Xmin 
    else:  
        I = (TailIndex/(TailIndex-1)) * (Xmin ** TailIndex) * (X ** (1-TailIndex))
    return I

@njit
def Pareto_CondExp_Int_Below(X,Xmin,TailIndex):
    '''
    Calculate int_Xmin^Xbar Z f(Z) dZ where f(Z) is the Pareto distribution with (Location,Shape) = (Xmin,TailIndex)
    IMPORTANT: need TailIndex > 1
    '''
    if X < Xmin: 
        I = 0   # Technically not a well-defined object
    else:
        I = (TailIndex/(TailIndex-1)) * (Xmin ** TailIndex) * ((Xmin**(1-TailIndex)) - (X ** (1-TailIndex)))
    return I

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

# ============================================================================================================
# Generic Dynamic Model Routines
# ============================================================================================================
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

# ------------------------------------------------------------------------------------------------------------
# Generalized Value Function Iterator
# ------------------------------------------------------------------------------------------------------------
def ValueFunctionIteration_2D_OneStep(ValueInit,ReturnSpace,StateGrid,StateDenseGrid,StochTransMatrix,discount):
    '''
    Generalized Routine for Value Function Iteration on a 2D State Space
    ********************************************************************
    ValueInit: np.array((nZ,nS)), function of (stochastic,nonstochastic) state.
    ReturnSpace: np.array((nZ,nS,nSDense)), a function of state today and state tomorrow implied by choices made today.
    StateGrid: np.array(nS), original grid of values for state
    StateDenseGrid: np.array(nSDense), grid of values for state with more gridpoints
    StochTransMatrix: np.array((nZ,nZ)), transition matrix
    discount: either np.float() or np.array() conformable for multiplication with np.array((nZ,nS)) and result being np.array((nZ,nS)), discount factor
    '''
    # Preallocate
    ValueNext,iPolicy = np.zeros(ValueInit.shape),np.zeros(ValueInit.shape,dtype='uint32')
    
    # Expectations
    EValue = StochTransMatrix @ ValueInit

    # Iterate
    nZ = ValueInit.shape[0]

    for iZ in range(nZ):

        # Interpolation
        Evalue_i = approx_1D(EValue[iZ],StateDenseGrid,StateGrid)

        # Return Function with broadcasting
        Return = ReturnSpace + discount * Evalue_i[:,np.newaxis]

        # Maximization Step 
        iPolicy[iZ] = np.nanargmax(Return,axis=0)
        ValueNext[iZ] = np.nanmax(Return,axis=0)        

    return ValueNext,iPolicy

def ValueFunctionIteration_3D_OneStep(ValueInit,ReturnSpace,StateGrid1,StateGrid2,StateDenseGrid1,StateDenseGrid2,nGrid1,nGrid2,StochTransMatrix,discount,SearchRestrictor):
    '''
    Generalized Routine for Value Function Iteration on a 3D State Space
    ********************************************************************
    ValueInit: np.array((nZ,nS1*nS2)), function of (stochastic,nonstochastic1,nonstochastic2) state, with values on 
               dimension 1 defined by combvec() (IMPORTANT: else interpolation is incorrect!) 
    ReturnSpace: np.array((nZ,nS1*nS2,nSDense1*nSDense2)), a function of state today and state tomorrow implied by choices made today.
    StateGrid1: np.array(nS1), original grid of values for state 1
    StateGrid2: np.array(nS2), original grid of values for state 2
    StateDenseGrid1: np.array(nSDense1), grid of values for state with more gridpoints
    StateDenseGrid1: np.array(nSDense1), grid of values for state with more gridpoints
    StochTransMatrix: np.array((nZ,nZ)), transition matrix
    nGrid1,nGrid2: np.uint32(), np.uint32(), aliases for nS1,nS2
    discount: either np.float() or np.array() conformable for multiplication with np.array((nZ,nS)) and result being np.array((nZ,nS)), discount factor
    SearchRestrictor: np.array((nZ,nS1*nS2,nSDense1*nSDense2)), array of 0/1 that defines where the max searches. Can help with speed by reducing 
                      the number of values over which the max operator searches.  
    '''
    # Preallocate
    ValueNext,iPolicy = np.zeros(ValueInit.shape),np.zeros(ValueInit.shape)
    
    # Expectations
    EValue = StochTransMatrix @ ValueInit

    # Iterate
    nZ = ValueInit.shape[0]

    for iZ in range(nZ):

        # Interpolation
        Evalue_i = approx_2D(EValue[iZ],StateDenseGrid1,StateDenseGrid2,StateGrid1,StateGrid2,nGrid1,nGrid2)

        # Return Function with broadcasting
        Return = ReturnSpace + discount * Evalue_i[:,np.newaxis]
        Return[1-SearchRestrictor] = np.NaN

        # Maximization Step: Grid Search 
        iPolicy[iZ] = np.nanargmax(Return,axis=0)
        ValueNext[iZ] = np.nanmax(Return,axis=0)        

    return ValueNext,iPolicy
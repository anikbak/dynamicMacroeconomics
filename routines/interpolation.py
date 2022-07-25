##############################################################################################################
# Routines Dedicated to Interpolation
##############################################################################################################
import numpy as np 
from numba import njit, guvectorize, cfunc

# ------------------------------------------------------------------------------------------------------------
# Grid Construction
# ------------------------------------------------------------------------------------------------------------
def combvec_2D(x1,x2,n1,n2):
    '''
    Construct state space compatible with linear interpolation operators below
    '''
    X1 = np.kron(x1,np.ones(n2))
    X2 = np.tile(x2,(1,n1)).squeeze()
    return np.array([X1,X2])

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

@njit("float64[:,:](float64[:],uint32,uint32)") 
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

@njit("Tuple((uint32[:],uint32[:],float64[:]))(float64[:], float64[:])")
def interpolate_coord_njit(x,xq):
    """
    Adapted from Adrien's notebook (Auclert et al 2021) and compiled via njit to make it 
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

########################################################################################################################
# ROUTINES TO SOLVE HOUSEHOLD PROBLEMS
########################################################################################################################
import numpy as np 
from numba import njit,cfunc
from routines.interpolation import * 

# ------------------------------------------------------------------------------------------------------------
# Generalized One-Asset Income Fluctuation Problem Block: Endogenous Gridpoint Method Steps
# ------------------------------------------------------------------------------------------------------------
@njit("UniTuple(float64[:,:],3)(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64,float64,float64,float64)") 
def EGMStep(mu_next,a_grid,e_grid,nl_inc_grid,eTrans,pbeta,eis,r_next,w):
    '''
    One step of a one-dimensional EGM Step with CRRA utility and inelastic labor supply
    ***********************************************************************************
    Consider a household whose assets live on a grid a_grid of dimension (nA,). 

    mu_next: np.ndarray((nE,nA))
        marginal utility next period
    a_grid: np.ndarray((nE,nA))  
        each row is the asset grid
    e_grid: np.ndarray((nE,nA))
        each column is the labor efficiency shock
    nl_inc_grid: np.ndarray((nE,nA))
        net non-labor income in the current period
    eTrans: np.ndarray((nE,nE))
        Transition matrix, eTrans[i,j] = Pr(e'=e[j] | e=e[i])
    pbeta: float64
        discount factor
    eis: float64
        elasticity of intertemporal substitution (= 1/coefft relative risk aversion)
    r_next: float64
        net interest rate between t and t+1
    w: float64 
        wage rate at date t
    
    '''

    # Dimensions
    nZ,nE = mu_next.shape

    # Expected Marginal utility
    EMU_next = eTrans @ mu_next

    # Evaluate current marginal utility and consumption using Euler equation
    MU_today = pbeta * (1+r_next) * EMU_next
    cons_next_grid = MU_today**(-eis)

    # Cash on Hand
    cash_on_hand = (1+r_next)*a_grid + w*e_grid + nl_inc_grid
    
    # Current assets as a function of current cash on hand 
    a_grid_today = np.zeros((nZ,nE))
    for iZ in range(nZ):
        a_grid_today[iZ] = interpolate_y_cfunc(cons_next_grid+a_grid,cash_on_hand,a_grid)
    
    a_grid_today = np.maximum(a_grid_today,a_grid[0][0])
    cons = cash_on_hand - a_grid_today

    return MU_today,cons,a_grid_today

@cfunc("UniTuple(float64[:,:],4)(float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64[:,:],float64,float64,float64,float64,float64,float64,float64,float64)") 
def EGMStep_EndogLabor(mu_next,a_grid,e_grid,nl_inc_grid,eTrans,pbeta,eis,frisch,ndisutil,r_next,w,ntaxrate):
    '''
    One step of a one-dimensional EGM Step with CRRA utility and inelastic labor supply
    ***********************************************************************************
    Consider a household whose assets live on a grid a_grid of dimension (nA,). 

    mu_next: np.ndarray((nE,nA))
        marginal utility next period
    a_grid: np.ndarray((nE,nA))  
        each row is the asset grid
    e_grid: np.ndarray((nE,nA))
        each column is the labor efficiency shock
    nl_inc_grid: np.ndarray((nE,nA))
        net non-labor income in the current period
    eTrans: np.ndarray((nE,nE))
        Transition matrix, eTrans[i,j] = Pr(e'=e[j] | e=e[i])
    pbeta: float64
        discount factor
    eis: float64
        elasticity of intertemporal substitution (= 1/coefft relative risk aversion)
    frisch: float64
        frisch elasticity of labor supply
    ndisutil: float64
        disutility of labor relative to marginal utility of cons.
    r_next: float64
        net interest rate between t and t+1
    w: float64 
        wage rate at date t
    ntaxrate: float64
        net labor income tax rate
    '''
    
    # Dimensions
    nZ,nE = mu_next.shape

    # Expected Marginal utility
    EMU_next = eTrans @ mu_next

    # Current marginal utilities
    MU_today = pbeta * (1+r_next) * EMU_next

    # Consumption, labor on next grid
    cons_next_grid = MU_today**(-eis)
    n_next_grid = (w*(1-ntaxrate) * MU_today / ndisutil)**frisch

    # Interpolation to get today's assets
    what = w * (1-ntaxrate)
    NR = cons_next_grid + a_grid - nl_inc_grid - what*e_grid*n_next_grid
    DR = (1+r_next)*a_grid
    c = interpolate_y_cfunc(NR,DR,cons_next_grid) 
    n = interpolate_y_cfunc(NR,DR,n_next_grid)
    a = (1+r_next)*a_grid + nl_inc_grid + what*e_grid*n - c

    indices_binds = np.nonzero(a<a_grid[0][0])
    a[indices_binds] = a_grid[0][0]
    
    # Solve for the optimal MU when constraint binds: system of two equations
    if len(indices_binds[0])>0 or len(indices_binds[1])>0:
        for i in range(len(indices_binds[0])):
            ix,iy = indices_binds[0],indices_binds[1]
            nl_inc = nl_inc_grid[ix][iy] + (1+r_next)*a[ix][iy] - a[0][0]
            MU_ini = MU_today[ix][iy]
            c[ix][iy],n[ix,iy] = StaticLaborSupplyFocSolve(MU_ini,w,ntaxrate,eis,ndisutil,frisch,nl_inc)

    return c,a,n

@cfunc('float64(float64,float64,float64,float64,float64,float64,float64,uint32,float64)',nopython=True)
def StaticLaborSupplyFocSolve(MU0,w,ntaxrate,eis,ndisutil,frisch,nl_inc,maxit=100,tol=1e-6):
    '''
    given mu_c, w, ntaxrate solve for c,n at each point of state space from static foc. That is, 
    solve the problem

    max_{c,n} k_c * c**(1-1/eis) - k_n * ndisutil*(n**(1+1/frisch)) st c = (1-ntaxrate)*w*n + nl_inc_grid

    Two equations: 
    **************
    c = (1-ntaxrate)*w*n + nl_inc_grid
    (1-ntaxrate)*w*(c**(-1/eis)) = ndisutil*(n**(1/frisch))

    Use an iteration scheme on the second foc. Following Auclert et al (2021), solve in log MU space.
    * Guess log(u'(c)) (note: essentially guessing lagrange multiplier on budget constraint!)
    * Can compute C,N given this 
    * Can check residual in budget constraint 

    '''
    logMU = np.log(MU0)
    what = w*(1-ntaxrate)

    for it in range(maxit):
        C,N = np.exp(logMU*(-eis)),(what * np.exp(logMU) / ndisutil) ** frisch
        res = C - what*N - nl_inc
        err = abs(res)
        if err < tol:
            break 
        else:
            logMU -= res/(-eis*C - w*frisch*N)

    else: 
        raise ValueError(f"constrained hh problem not solved after {maxit} iterations")

    C,N = np.exp(logMU*(-eis)),(what * np.exp(logMU) / ndisutil) ** frisch
    return C,N

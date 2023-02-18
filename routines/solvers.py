import numpy as np
from numpy.random import default_rng

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
    while (np.max(np.abs(y))>tol) & (it<=maxit):

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

def SimpleNewtonSolverBroyden(f,xinit,JacRoutine=None,yinit=None,tol=1e-5,maxit=1000,backtrack_c=0.5,report=True):
    '''
    Line search solver for root of a multidimensional function with Broyden updating for the Jacobian
    *************************************************************************************************
    Essentially from the routine in Auclert et al (2021). 
    Uses backtracking to avoid basin problems (somewhat). 
    Main issue: relies on potentially expensive Jacobian evaluations (sped up a bit by Broyden updates). 
    
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
    while (np.max(np.abs(y))>tol) & (it<=maxit):

        # Check if converged
        if np.max(np.abs(y))<=tol:            
            return x,y

        if report==True: 
            print(f'iteration {iter}, current max abs error: {np.max(np.abs(y))}, current mean abs error: {np.mean(np.abs(y))}')

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
    Npop_current = population.shape[0]
    fitness = np.array([func(population[i]) for i in range(Npop_current)])
    population = population[np.argsort(fitness),:]
    fitness = fitness[np.argsort(fitness)]

    # Step 3: Deaths (kill least fit agents)
    Nsurvivors = int( (1-death_rate)*Npop )
    population = population[:Nsurvivors,:]
    fitness = fitness[:Nsurvivors]
    Npop_current = population.shape[0]

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
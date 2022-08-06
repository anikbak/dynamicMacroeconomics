import numpy as np
from quantecon.markov import tauchen,rouwenhorst

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
    * typevecZ: list with length = # exog variables with each element either "tauchen" or "rouwenhorst"

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
        TransTotal = TransZ[StatenamesZ[0]]

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

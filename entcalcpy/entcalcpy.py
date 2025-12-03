"""
entcalcpy — Python package for computing geometric entanglement.

This package provides functions for calculating lower and upper bounds
of the geometric entanglement for pure and mixed quantum states.
The bounds are obtained using semidefinite programming (SDP) and other
optimization techniques, allowing efficient estimation of the geometric
entanglement even in noisy or high-dimensional systems.

Main features:
- Upper bound estimation for multipartite and bipartite states
- Multiple methods for computing lower bounds
- Works with QuTiP quantum objects
- Simple API for rapid prototyping and research use

Example:
    >>> import qutip
    >>> import entcalcpy as en
    >>> rho = qutip.rand_dm(8)
    >>> en.uppersame(rho, [2, 2, 2])
    >>> en.ge_mixed_gr(rho, [2, 2, 2])

Authors:
    Piotr Masajada
    Aby Philip
    Alexander Streltsov
"""



import cvxpy as cp
import qutip
import numpy as np
from scipy.linalg import sqrtm
from qutip import qeye, tensor, Qobj
import copy
import warnings
import itertools
import scipy
from itertools import combinations_with_replacement
from collections import Counter

    


def random_haar_state(dim):
    """
    Generate a random pure quantum state distributed according to the Haar measure.

    This function creates a random complex vector of length `dim` where both the real
    and imaginary parts are drawn from a standard normal distribution. The vector is
    then normalized to have unit norm, ensuring it represents a valid pure quantum state.
    The resulting state is returned as a QuTiP ``Qobj``.

    Parameters
    ----------
    dim : int
        The dimension of the Hilbert space (i.e., the size of the quantum state vector).

    Returns
    -------
    qutip.Qobj
        A normalized quantum state vector (pure state) drawn from the Haar measure.

    Notes
    -----
    - The Haar measure ensures that all pure states are generated with equal probability.
    - This is commonly used in simulations of random quantum states and benchmarking
      quantum algorithms.

    Examples
    --------
    >>> psi = random_haar_state(4)
    >>> psi.isket
    True
    >>> psi.norm()
    1.0
    """
    z = np.random.randn(dim) + 1j * np.random.randn(dim)
    
    z /= np.linalg.norm(z)
    return qutip.Qobj(z)



def ge_pure(rho,dim,sdpaccuracy=10**(-14),itera=5000, solversdp="SCS"):
    """
    Return the geometric entanglement of a pure state `rho`.

    `rho` should be given in ket form. `dim` is a list of dimensions
    of each subsystem. `n` is the number of subsystems.

    Parameters
    ----------
    rho : array_like
        A 1D array representing a pure state in ket form.

    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    
    sdpaccuracy : int or str, optional. Default is 1e-14
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).

    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.

    solversdp : str, optional
        SDP solver to use ('SCS', 'MOSEK'). Default is 'SCS'.
        See the CVXPY documentation for solver detail.

    Returns
    -------
    list
        A list containing:
        - The geometric entanglement of `rho` (float).
        - The numerical error of the result (float).
        
        
    Notes
    -----
    - For qubit-qubit and qubit-qutrit our function gives exact solutions. The only error is a numerical error connected to SDP.
    - For a high-precision computations we recommend using MOSEK as a solver.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    
    >>> GHZ.ge_pure(GHZ,[2,2,2],3)
    [0.4999999999999999, 0.0] The only error is connected with SDP precision
    """
    
    dim=copy.deepcopy(dim)
    if not isinstance(dim, list):
        raise Exception("dim should be a list")
    n=len(dim)
    #Checking correctness of inputs
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    producta=1
    for ih in dim:
        producta*=ih
    rst=qutip.Qobj(rho)
    if len(rst.full())!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")
    
    if abs(1-rst.norm())>10**(-8):
        warnings.warn("Warning! State is not normalized. The result in that scenario is not geometric entanglement.")
    if not rst.isket:
        raise Exception("State should be given in ket form.")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
       #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-14,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-14,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-14,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 900,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-9,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-9,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-9,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 600,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 400,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")
    rst.dims=[dim,[1]*len(dim)]
        
    product=1
    max_index=dim.index(max(dim))
    dim.remove(max(dim))
    li=list(range(n))
    
    li.remove(max_index)
    
    rstpro = rst.proj()
    rpt = rstpro.ptrace(li)
    rh = rpt.full()
    
    for ih in dim:
        product*=ih
        
    
    sigma = cp.Variable((product,product),complex=True)
    
    objective=cp.real(cp.trace(rh@sigma))
    constraint=[sigma>>0,cp.trace(sigma)==1]
    for i in range(len(dim)):       
        constraint+=[cp.partial_transpose(sigma,dim,i)>>0]
    pp=cp.Problem(cp.Maximize(objective),constraint)
    if solversdp=="SCS":        
        solution=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
    if solversdp=="MOSEK":                                                     #Solver setting. If you want to amend setting in a different 
        solution=pp.solve(solver="MOSEK",mosek_params=params)                      #way that is allowed in function call, do it here

    sig=sigma.value
    ww=np.real(np.linalg.eigvals(sig))
    maxww=max(ww)
    ep=abs(maxww-1)
    if product<=6 and n==3:               #PPT equivalent to separability for these states
        accura=0
        
    else:
        if n==3:
            acc=4*(n-2)*np.sqrt(ep)             #Formula for error shown in the paper appendix
            accura=float(acc)
        
    
    if accura>0.01:
        if abs(sum(sig.flatten())-np.trace(sig))<10**(-10):
            
            sig_dia=np.diag(np.diag(sig))
            acc=np.trace(sig_dia@rh)
            accura=float(abs(solution-acc))
    sol=1-solution
    if accura>0.01 and product>6:
        warnings.warn("Warning! entcalcpy could not find good approximation bound.")
        return [sol,accura]
    return [sol,accura]
    



def vec(x):
    """
    Vectorize a QuTiP operator given in matrix form.

    This function takes a QuTiP `Qobj` representing
    a quantum operator in matrix form and returns its vectorized form
    (flattened into a 1D array) using column-stacking convention.

    Parameters
    ----------
    op : qutip.Qobj
        Operator to be vectorized. Must be a square matrix.

    Returns
    -------
    qutip.Qobj
        1D complex vector obtained by stacking the columns of the input
        matrix on top of each other.

    Notes
    -----
    This is equivalent to the `vec` operation commonly used in
    quantum information theory, where `vec(|i⟩⟨j|) = |i⟩ ⊗ |j⟩`.
    """
    
    xx=x.trans()
    xxx=xx.full()
    x=xxx.flatten()
    xx=np.transpose(np.array([x]))
    a=qutip.Qobj(xx)
    
    return(a)

def ge_mixed_sm (state,dim,sdpaccuracy=10**(-8),itera=5000, solversdp="SCS"):#faster
    #x=np.array([1,0,0])
    """
    Computes a lower bound on the geometric entanglement of a mixed quantum state using a purity-entanglement complementarity relation.

    Parameters
    ----------
    state : array_like
        A 2D array representing a mixed state (density matrix).
    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).
    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.
    solversdp : str, optional
        SDP solver to use ('SCS', 'MOSEK'). Default is 'SCS'.
        See CVXPY documentation for solver detail.

    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.   
    Notes
    -----
    - These function computes lower bound using purity-entanglement complementary relation. This is one among two functions which uses this relation. This
    function is faster but usually gives worse bound.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> ge_mixed_sm(rho,[2,2,2])
    [0.3025200020841136, 'Accurate']
    """
    #Checking correctness of inputs
    if not isinstance(dim, list):
        raise Exception("dim shoul be a list")
    n=len(dim)
    
    dim=copy.deepcopy(dim)
    
    
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    state=qutip.Qobj(state)
    state=copy.deepcopy(state)
    if not state.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = state.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")

    if abs(state.tr()-1)>10**(-10):
        raise Exception("State should be normalized")
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")
                        
    producta=1
    for ih in dim:
        producta*=ih
    if len(state.full())!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")

    
    #Main part of the function
    rho=state.sqrtm()
    rho.dims=[dim,dim]
    ket=qutip.Qobj(vec(rho))                          #Purification
    ket.dims=[[producta]+dim,[1]*(n+1)]

    stp=ket.ptrace(list(range(n)))

    
    pro=int(producta**(2)/dim[-1])
    
    sta=stp.full()
    del dim[-1]
    product=1
    for ih in dim:
        product*=ih
    
    X=cp.Variable((pro,pro),hermitian=True)
    con=[X>>0,cp.partial_trace(X,[producta,product],1)==np.eye(producta)]  
    for i in range(n):       
        con+=[cp.partial_transpose(X,[producta]+dim,i)>>0]       #PPT relaxation
    obj=cp.trace(X@sta)
    pp=cp.Problem(cp.Maximize(cp.real(obj)),con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol,"Inaccurate"]
        else:
            return [1-sol,"Accurate"]
    


def ge_mixed_gr (state,dim,sdpaccuracy=10**(-8),itera=5000, solversdp="SCS"):#faster
    
    """
    Computes a lower bound on the geometric entanglement of a mixed quantum state using a purity-entanglement 
    complementary relation with higher dimensional matrices.

    Parameters
    ----------
    state : array_like
        A 2D array representing a mixed state (density matrix).
    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).
    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000. Relevant for SCS solver
    solversdp : str, optional
        SDP solver to use (e.g., 'SCS', 'MOSEK'). Default is 'SCS'. See CVXPY documentation for available solvers.

    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.
    Notes
    -----
    - These function computes lower bound using purity-entanglement complementary relation. This is one among two functions which uses this relation. This
    function can be much slower but usually gives better bound.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> ge_mixed_gr(rho,[2,2,2])
    [0.30252000211612345, 'Accurate']
    """
    if not isinstance(dim, list):
        raise Exception("dim shoul be a list")
    n=len(dim)
    state=qutip.Qobj(state)
    state=copy.deepcopy(state)
    #Checking correctness of inputs
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not state.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = state.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
   
    
    if abs(state.tr()-1)>10**(-10):
        raise Exception("State should be normalized")
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_NUM_COR": 50,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")
    #Dimensionality check
    producta=1
    for ih in dim:
        producta*=ih
    if len(state.full())!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")

    
    #Main part of the function starts here
    rho=state.sqrtm()                         
    rho.dims=[dim,dim]
    ket=qutip.Qobj(vec(rho))                   #Purification
    ket.dims=[[producta]+dim,[1]*(n+1)]
   

    stp=ket.proj()


    pro=producta**(2)
    
    sta=stp.full()
    
    
    
    X=cp.Variable((pro,pro),hermitian=True)
    con=[X>>0,cp.partial_trace(X,[producta,producta],1)==np.eye(producta)]     
    for i in range(n+1):       
        con+=[cp.partial_transpose(X,[producta]+dim,i)>>0]            #PPT relaxation
    obj=cp.trace(X@sta)
    pp=cp.Problem(cp.Maximize(cp.real(obj)),con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision.")
            return [1-sol,"Inaccurate"]
        else:
            return [1-sol,"Accurate"]



def ge_mixed_ra_gr (kets,ps,dim,sdpaccuracy=10**(-8),itera=5000, solversdp="SCS"):
    
    """
    Computes a lower bound on the geometric entanglement of a mixed quantum state using a purity-entanglement 
    complementary relation with higher dimensional matrices. Contrary to ge_mixed_gr this function accepts input
    as a ensemble of orthogonal kets not a density matrix.
    Because of this it can find a more efficient purification.

    Parameters
    ----------
    kets : array_like
        A list of kets (arrays representing a pure state). Kets must be orthogonal.
    ps : list of floats.
        A list of non-negative floats representing probability of given ket in state ensemble. 
    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).
    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.
    solversdp : str, optional
        SDP solver to use (e.g., 'SCS', 'MOSEK'). Default is 'SCS'. See CVXPY documentation for available solvers.

    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.
        
    Notes
    -----
    - These function computes lower bound using purity-entanglement complementary relation. This is one among two functions which uses this relation. This
    function can be much slower but usually gives better bounds.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> pslist=[0.6,0.4]
    >>> ge_mixed_ra_gr([GHZ,W],pslist,[2,2,2])
    [0.3025200021131412, 'Accurate']
    """
    
    n=len(dim)    
    #Checking correctness of inputs
    if not isinstance(dim, list):
        raise Exception("dim shoul be a list")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    
    
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")
    
    producta=1
    for ih in dim:
        producta*=ih

    
    
          
    
    ke=[]
    rank=len(kets)
    for i in range(len(kets)):
        ke.append(qutip.Qobj(kets[i]))
        ke[i].dims=[dim,[1]*n]


    for i in range(rank):
        for j in range(i+1, rank):
            overlap = ke[i].overlap(ke[j])  # <i|j>
            if abs(overlap) > 10**(-14):
                raise Exception("kets should be orthogonal")             #Only works for orthogonal states
    nk=len(ke[0].full())
    for i in range(rank):
        
        if not abs(ke[i].norm()-1)<10**(-14):
            raise Exception("States should be normalized")
        if len(ke[0].full())!=nk:
            raise Exception("States should have the same dimensions")

    if nk!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")
    if abs(sum(ps)-1)>10**(-14):
        raise Exception("probabilities does not sum up to 1")

    if min(ps)<=-10**(-14):
        raise Exception("probabilities should be greater than 0")

    
    
    #Main part of the function
    stb=qutip.Qobj(np.zeros(producta*rank,dtype=complex))
    stb.dims=[[rank]+dim,[1]*(n+1)]
    for i in range(rank):
        stb+=np.sqrt(ps[i])*qutip.tensor(qutip.basis(rank,i),ke[i])       #faster purification
    
    
    
    

    stp=stb.proj()

    
    
    
    sta=stp.full()
    
    
    pro=producta*rank
    X=cp.Variable((pro,pro),hermitian=True)
    con=[X>>0,cp.partial_trace(X,[rank,producta],1)==np.eye(rank)]
    for i in range(n+1):       
        con+=[cp.partial_transpose(X,[rank]+dim,i)>>0]                         #PPT relaxation
    obj=cp.trace(X@sta)
    pp=cp.Problem(cp.Maximize(cp.real(obj)),con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision.")
            return [1-sol,"Inaccurate"]
        else:
            return [1-sol,"Accurate"]




def ge_mixed_ra_sm (kets,ps,dim,sdpaccuracy=10**(-8),itera=5000, solversdp="SCS"):
    
    """
    Computes a lower bound on the geometric entanglement of a mixed quantum state using a purity-entanglement 
    complementary relation with higher dimensional matrices. Contrary to ge_mixed_sm this function accepts input
    as a ensemble of orthogonal kets not a density matrix.
    Because of this it can find a more efficient purification.

    Parameters
    ----------
    state : array_like
        A 2D array representing a mixed state (density matrix).
    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).
    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.
    solversdp : str, optional
        SDP solver to use (e.g., 'SCS', 'MOSEK'). Default is 'SCS'. See CVXPY documentation for available solvers.

    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.

    Notes
    -----
    - These function computes lower bound using purity-entanglement complementary relation. This is one among two functions which uses this relation. This
    function is faster but usually gives worse bounds.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> pslist=[0.6,0.4]
    >>> ge_mixed_ra_sm([GHZ,W],pslist,[2,2,2])
    [0.3025200004028166, 'Accurate']
        
    """
    n=len(dim)
    producta=1
    dim=copy.deepcopy(dim)
    #Checking correctness of inputs
    if not isinstance(dim, list):
        raise Exception("dim shoul be a list")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    
    
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")

    for ih in dim:
        producta*=ih
    
    ke=[]
    rank=len(kets)
    for i in range(len(kets)):
        ke.append(qutip.Qobj(kets[i]))
        ke[i].dims=[dim,[1]*n]

    for i in range(rank):
        for j in range(i+1, rank):
            overlap = ke[i].overlap(ke[j])  # <i|j>
            if abs(overlap) > 10**(-14):
                raise Exception("kets should be orthogonal")                               #Only works for orthogonal states



    
    nk=len(ke[0].full())
    for i in range(rank):
        if not abs(ke[i].norm()-1)<10**(-14):
            raise Exception("States should be normalized")
        if len(ke[0].full())!=nk:
            raise Exception("States should have the same dimensions")

    if nk!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")
    if abs(sum(ps)-1)>10**(-14):
        raise Exception("probabilities does not sum up to 1")

    if min(ps)<=-10**(-14):
        raise Exception("probabilities should be greater than 0")

    
    #Main part of the function
    stb=qutip.Qobj(np.zeros(producta*rank,dtype=complex))
    stb.dims=[[rank]+dim,[1]*(n+1)]
    for i in range(rank):
        stb+=np.sqrt(ps[i])*qutip.tensor(qutip.basis(rank,i),ke[i])         #faster purification
    
    
    
    

    stp=stb.proj()

    
    
    
    sta=stb.ptrace(list(range(n)))
    sta=sta.full()
    
    pro=int(producta*rank/dim[-1])
    product=int(producta/dim[-1])
    del dim[-1]
    X=cp.Variable((pro,pro),hermitian=True)
    con=[X>>0,cp.partial_trace(X,[rank,product],1)==np.eye(rank)]
    for i in range(n):       
        con+=[cp.partial_transpose(X,[rank]+dim,i)>>0]                          #PPT relaxation
    obj=cp.trace(X@sta)
    pp=cp.Problem(cp.Maximize(cp.real(obj)),con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision.")
            return [1-sol,"Inaccurate"]
        else:
            return [1-sol,"Accurate"]



def geppt(rho,dim,sdpaccuracy=10**(-8),itera=5000,solversdp="SCS"):
    
    """

    Function which computes the lower bound of the geometric entanglement of the state rho using fidelity relation and PPT relaxation.
    

    Parameters
    ----------
    rho : array_like
        A 2D array representing a mixed state.
    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).
    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.
    solversdp : str, optional
        SDP solver to use (e.g., 'SCS', 'MOSEK'). Default is 'SCS'. See CVXPY documentation for available solvers.
    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.
    
    Notes
    -----
    - These function computes lower bound using relation E(\rho)=1-max(F(rho,sigma)), where maximization is done over
    separable states. In this function we relax separability condition to PPT obtaining a lower bound.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> geppt(rho,[2,2,2],solversdp="MOSEK")
    0.30251626223193795
    """
    n=len(dim)
    #Checking correctness of inputs
    if not isinstance(dim, list):
        raise Exception("dim shoul be a list")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    rho=qutip.Qobj(rho)
    if not rho.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = rho.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
   
    
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    
    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")                    
    producta=1


    
    for ih in dim:
        producta*=ih
    if len(rho.full())!=producta:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")

    
    #Main part of the function
    rho=qutip.Qobj(rho)
    rho.dims=[dim,dim]
    rho=copy.deepcopy(rho)
    pro=1
    n=len(dim)
    for ih in dim:
        pro*=ih
    zer=np.zeros([pro,pro],dtype=complex)
    A=cp.bmat([[zer,np.eye(pro)/2],
               [np.eye(pro)/2,zer]])
    
    
    sigma=cp.Variable((pro,pro),hermitian=True)
    con=[sigma>>0,cp.trace(sigma)==1]
    for i in range(n):       
        con+=[cp.partial_transpose(sigma,dim,i)>>0]
    X=cp.Variable((pro,pro),complex=True)
    B=cp.bmat([[rho.full(),X],[cp.conj(cp.transpose(X)),sigma]])         #We compute fidelity between rho and closest PPT state sigma
    con+=[B>>0]
    obj=cp.Maximize(cp.real(cp.trace(A@B)))
    pp=cp.Problem(obj,con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision.")
            return [1-sol**(2),"Inaccurate"]
        else:
            return [1-sol**(2),"Accurate"]

def symmetric_subspace_basis(d, k):
    

    # All integer multisets of length k from {0,...,d-1}
    basis_states = list(combinations_with_replacement(range(d), k))
    sym_vectors = []

    for state in basis_states:
        # Generate all distinct permutations
        perms = set(itertools.permutations(state))
        ket_sum = sum(qutip.tensor([qutip.basis(d, i) for i in p]) for p in perms)
        ket_sym = (ket_sum.unit())  # normalize
        sym_vectors.append(ket_sym)

    return sym_vectors

def symmetric_projector_from_basis(d, k):
    basis = symmetric_subspace_basis(d, k)
    P = sum([v.proj() for v in basis])
    return P

def gekppt(rho,dim,k,sdpaccuracy=10**(-8),itera=5000,solversdp="SCS"):
    """
    Function which computes the lower bound of the geometric entanglement of the state `rho` using fidelity relation,
    PPT relaxation and k-symmetric extensions.   
       Parameters
    ----------
    rho : array_like
        A 2D array representing a mixed bipartite state.

    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    k : int
        Number of k-symmetric extensions. 
        For a bipartite state ρ^(AB), this specifies the number of copies of subsystem B.
        For example, if k = 2, the extended state is ρ^(ABB').
        For multipartite states, k defines the number of additional copies of each subsystem 
        beyond the first one. For example, for a tripartite state ρ^(ABC) and k = 4, 
        the extended state is ρ^(ABCB'C'B''C'').

    sdpaccuracy : int or str, optional. Default is 1e-8
        Precision control for the solver.
        
        - If an **int** is given, it is interpreted as the number of digits
          of numerical precision to use. Recommended for SCS solver.
        - If a **str**, it must be one of (recommended for MOSEK):
          
          * `"high"` – uses strict solver tolerances (slow but accurate),
          * `"medium"` – balanced setting,
          * `"low"` – loose tolerances (fast but less accurate).

    itera : int, optional
        Maximum number of iterations for the SDP. Default is 5000.
    solversdp : str, optional
        SDP solver to use (e.g., 'SCS', 'MOSEK'). Default is 'SCS'. See CVXPY documentation for available solvers.
    Returns
    -------
    list
    A list containing:
    - The lower bound of the geometric entanglement of `rho` (float).
    - A string indicating whether the result meets the required precision:
        - "Accurate" → the solver states that the SDP is solved with the desired precision. See solver specification for more information.
        - "Inaccurate" → the solver stopped before reaching optimum with desired precision. See solver specifications for more information.

    Notes
    -----
    - These function computes lower bound using relation E(\rho)=1-max(F(rho,sigma)), where maximization is done over
    separable states. In this function we relax separability condition to PPT together with k-symmetric extendibility obtaining a lower bound.
    - For a high-precision computations we recommend using MOSEK as a solver.
    - If the solution is inaccurate and the solver used is SCS, the outcome might not be a lower bound. There are 3 ways to proceed.
    1) Increase the itera, since more iteration may lead to accurate outcome.
    2) Decrease required precision. With worse precision solver might give accurate outcome and then entcalpy will be able to provide lower bound.
    3) Change solver to MOSEK.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> gekppt(rho,[2,2,2],2,solversdp="MOSEK")
    0.30251624822853207
    
    """
    rho=qutip.Qobj(rho)
    
    rho=copy.deepcopy(rho)
    #Checking correctness of inputs
    n=len(dim)
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not rho.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = rho.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    
    if itera<0:
        raise Exception("itera must be an integer greater than 0")
    if not isinstance(itera, int):
        raise Exception("itera must be an integer greater than 0")
    if n<=1:
        raise Exception("System must be a compound quantum system.")
    if not (solversdp!="MOSEK" or solversdp!="SCS"):
        raise Exception("Only MOSEK and SCS are compatible with entcalcpy.")
    if not isinstance(k, int):
        raise Exception("k must be an integer greater than 0")
    if k<=0:
        raise Exception("k must be an integer greater than 0") 

    #Setting MOSEK parameters
    mosek_params_high = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-12,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    }

    mosek_params_medium = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 300,
    }

    mosek_params_low = {
        "MSK_IPAR_NUM_THREADS": 1,
        "MSK_IPAR_PRESOLVE_USE": 0,
        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-6,
        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-6,
        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-6,
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 200,
    }

    
    precision_settings = {
        "high": mosek_params_high,
        "medium": mosek_params_medium,
        "low": mosek_params_low,
    }
    if isinstance(sdpaccuracy, float):
        if sdpaccuracy<0:
            raise Exception("sdpaccuracy must be a float greater than 0")
        params = {
            "MSK_IPAR_NUM_THREADS": 1,
            "MSK_IPAR_PRESOLVE_USE": 0,
            "MSK_DPAR_INTPNT_TOL_REL_GAP": 10**(-sdpaccuracy-1),
            "MSK_DPAR_INTPNT_TOL_PFEAS": 10**(-sdpaccuracy-2),
            "MSK_DPAR_INTPNT_TOL_DFEAS": 10**(-sdpaccuracy-2),
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        }
        
    elif isinstance(sdpaccuracy, str):
        params = precision_settings[sdpaccuracy]
        if solversdp=="SCS":
            if sdpaccuracy=="high":       #SCS only accepts floats as accuracy
                sdpaccuracy=10**(-8)
            if sdpaccuracy=="medium":
                sdpaccuracy=10**(-6)
            if sdpaccuracy=="low":
                sdpaccuracy=10**(-4)
    else:
        raise ValueError("sdpaccuracy must be a float or one of {'high','medium','low'}")


    
    pro=1
    for i in dim:
        pro*=i
    
    if len(rho.full())!=pro:
        raise Exception("Dimension of the system is not equal to product of dimensions of subsystem")

    
    
    #Main part of the function
    rho.dims=[dim,dim]
    zer=np.zeros([pro,pro],dtype=complex)
    A=cp.bmat([[zer,np.eye(pro)/2],
               [np.eye(pro)/2,zer]])
    
    
    
    con=[]
    if n==2:     #bipartite states
        if k<2:
            raise Exception("k should be greater or equal 2")
        sigma=cp.Variable((dim[0]*dim[1]**(k),dim[0]*dim[1]**(k)),hermitian=True)
        for il in range(1,k):
            
            if k-il-1>0:
                
                con+=[cp.partial_trace(cp.partial_trace(sigma,[dim[0],dim[1]**(il),dim[1],dim[1]**(k-il-1)],3),[dim[0],dim[1]**(il),dim[1]],1)==cp.partial_trace(sigma,[dim[0],dim[1],dim[1]**(k-1)],2)]       #k-symmetric extendibility condition
                
            else:
                
                con+=[cp.partial_trace(sigma,[dim[0],dim[1]**(k-1),dim[1]],1)==cp.partial_trace(sigma,[dim[0],dim[1],dim[1]**(k-1)],2)]       #k-symmetric extendibility condition

        con+=[cp.trace(cp.partial_trace(sigma,[dim[0],dim[1],dim[1]**(k-1)],2))==1]
        for i in range(k+1):
            con+=[cp.partial_transpose(sigma,[dim[0]]+[dim[1]]*(k),i)>>0]
        
        con+=[sigma>>0]      #We compute fidelity between rho and closest PPT and k-symmetric extendible state sigma
        
        X=cp.Variable((pro,pro),complex=True)
        B=cp.bmat([[rho.full(),X],[cp.conj(cp.transpose(X)),cp.partial_trace(sigma,[dim[0]*dim[1],dim[1]**(k-1)],1)]])
        con+=[B>>0]



        
    else:
        
        reszta=k%(n-1)
        kk=int(np.floor(k/(n-1)))
        ran=list(range(len(dim)))
        ranco=list(range(len(dim)))
        n_dim=dim+dim[1:]*kk+dim[n-reszta:]
        n_num=ran+ran[1:]*kk+ran[n-reszta:]
        add_di=[1]*(n-1)                   
        for i in range(k):
            add_di[i%(n-1)]+=1            #How many times is subsystem extended.
        
        n_pr=1
        for qw in n_dim:
            n_pr*=qw
        n_wi=1
        for qw in dim[1:]:
            n_wi*=qw
        n_number=len(n_dim)
        sigma=cp.Variable((n_pr,n_pr),hermitian=True)
        n_without=1
        for qw in n_dim[n:]:
            n_without*=qw
        ranco.reverse()
        
        for sy in range(n-1):    #We find indices that we want to keep, stored in lists zostaw and zostaw2
            
            for napot in range(1,add_di[sy]):
                ile=-1
                ilesys=0
                zostaw=[]
                zostaw2=[]
                czy=1
                wywal=[]
                
                while len(zostaw+zostaw2)<n:
                    ile+=1
                    
                    
                    if n_num[ile]!=ranco[sy] and czy==1:
                        zostaw.append(ile)
                    elif n_num[ile]!=ranco[sy] and czy==0:
                        wywal.append(ile)
                    else:
                        ilesys+=1
                        
                        if ilesys<=napot:
                            czy=0
                            wywal.append(ile)
                        if ilesys==napot+1:
                            zostaw2.append(ile)
                            if sy>0:
                                for isy in range(1,sy+1):
                                    zostaw2.append(ile+isy)
                    
                l_dim=1
                for iz in zostaw:             #Dimensions of the systems that we trace out and keep
                    l_dim*=n_dim[iz]
                m_dim=1
                for iz in zostaw2:
                    m_dim*=n_dim[iz]
                z_dim=1
                for iz in wywal:
                    z_dim*=n_dim[iz]
                if l_dim*m_dim*z_dim==n_pr:
                    con+=[cp.partial_trace(sigma,[l_dim,z_dim,m_dim],1)==cp.partial_trace(sigma,[pro,n_without],1)]       #k-symmetric extendibility condition
            

                else:
                    pozos=int(n_pr/(l_dim*m_dim*z_dim))
                    
                    con+=[cp.partial_trace(cp.partial_trace(sigma,[l_dim,z_dim,m_dim,pozos],3),[l_dim,z_dim,m_dim],1)==cp.partial_trace(sigma,[pro,n_without],1)]       #k-symmetric extendibility condition
                        
                         
        X=cp.Variable((pro,pro),complex=True)
        B=cp.bmat([[rho.full(),X],[cp.conj(cp.transpose(X)),cp.partial_trace(sigma,[pro,n_without],1)]])
        con+=[cp.trace(cp.partial_trace(sigma,[pro,n_without],1))==1]
        for i in range(n_number):
            con+=[cp.partial_transpose(sigma,n_dim,i)>>0]        #Plus PPT condition
        
                
      
        con+=[sigma>>0]      #We compute fidelity between rho and closest PPT and k-symmetric extendible state sigma
        
        
        con+=[B>>0]
    obj=cp.Maximize(cp.real(cp.trace(A@B)))
    pp=cp.Problem(obj,con)
    if solversdp=="SCS":        
        sol=pp.solve(solver="SCS",max_iters=itera,eps=sdpaccuracy)  
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision. Try increasing itera to prevent this. Lower bound mighnt not be a lower bound")
            return [1-sol**(2)-sdpaccuracy,"Inaccurate"]
        else:
            return [1-sol**(2)-sdpaccuracy,"Accurate"]
    if solversdp=="MOSEK":                                            #Solver setting. If you want to amend setting in a different
        sol=pp.solve(solver="MOSEK",mosek_params=params)                  #way that is allowed in function call, do it here
       
        if pp.status == "optimal_inaccurate":
            warnings.warn("Solution did not match with desired precision.")
            return [1-sol**(2),"Inaccurate"]
        else:
            return [1-sol**(2),"Accurate"]














def upperbip(rho,dim,iteramax=3000,dif=10**(-7),r=None,qs=None,sqs=None,dec=False):
    """
    Function which computes the upper bound of the geometric entanglement of the bipartite state rho.    
        Parameters
    ----------
    rho : array_like
        A 2D array representing a mixed state.

    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    

    iteramax : int, optional
        Maximum number of iterations of the algorithm. Default is 3000.

    dif : float, optional
        Parameter determining if the algorithm converged. Default is 10**(-7).
        
    r : int, optional
        Number of pure states in the separable decomposition of `rho`. 
        Default is `pro**2`, where `pro` are the subsystem dimensions of `rho`. 
        A recommended choice is `rank(rho)**2`.

    qs : list of float, optional
        Probabilities in the initial separable decomposition of `rho`.

    sqs : list of numpy.ndarray, optional
        Pure states (kets) in the initial separable decomposition of `rho`, 
        each given as a NumPy array.

    dec : bool, optional
        If True, the separable decomposition is also returned. 
        If False (default), only the bound is returned.

    Returns
    -------
    float or tuple
        If ``dec=False`` (default), returns the upper bound on the geometric 
        entanglement of `rho` as a float.  
        If ``dec=True``, returns a tuple ``(bound, decomposition)``, where 
        ``bound`` is the upper bound (float) and ``decomposition`` is the 
        separable decomposition (list of probabilities and pure states).

    Notes
    -----
    - These function computes upper bound using iteration alorithm. Each step of the algorithm finds a better
    separable decomposition of maximization variable.
    - Input state must be a bipartite state
    - Due to taking square roots and machine precision 10^(-16), accuracy of computation using upperbip is around 10^(-8). This accuracy is not
    included in the result, meaning that in principle upper bound computed by this function can be 10^(-8) lower then true value of the
    geometric entanglement.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> upperbip(rho,[4,2])  
    0.28918150279345456
    """
    
    rho=qutip.Qobj(rho)
    
    rho=copy.deepcopy(rho)
    #Checking correctness of inputs
    if abs(rho.tr()-1)>10**(-10):
        raise Exception("State should be normalized")
    if len(dim)!=2:
        raise Exception("System should be a bipartite quantum state")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not rho.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = rho.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    if len(rho.full())!=dim[0]*dim[1]:
        raise Exception("Dimension of the system is not equal to the product of dimensions of subsystem")
    if r==None:
        r=(dim[0]*dim[1])**(2)




    if (not isinstance(iteramax,int)) or iteramax<=0:
        raise ValueError("iteramax should be an integer greater than 0.")
    if not isinstance(r,int):
        raise ValueError("r should be an integer")
    if (not isinstance(dif,float)) or not (0<dif<1):
        raise ValueError("dif should be in interval (0,1)")
    if (qs is None and (not sqs is None)) or (sqs is None and not (sqs is None)):
        raise Exception("Both or none of the qs, sqs must be None.")
    if not (sqs is None) and r!=len(sqs.T):
        raise Exception("sqs and r should have the same number of elements")
    if not (qs is None) and r!=len(qs):
        raise Exception("qs and r should have the same number of elements")
    if r>(dim[0]*dim[1])**(2):
        raise Exception(f"r should not be greater than dim(rho)**(2)={(dim[0]*dim[1])**(2)}")
    if not (dec==True or dec==False):
        raise Exception("dec should be True or False")

    
    #Main part of the function
    rho.dims=[dim,dim]
    F=69
    itera=0
    Fp=0
    n=len(rho.full())
    eig=rho.eigenstates()
    pqp=eig[0]
    pqp = np.where(pqp < 10**(-15), 0.0, pqp)             #Handling numerical errors in eigendecomposition
    licznik = np.count_nonzero(pqp > 10**(-15))
    pqp=pqp/sum(pqp)
    if r<licznik:
        raise Exception(f"Number r of vectors in decomposition is too small to represent rho. The number of non-zero eigenvalues of rho is {licznik}")
    p=np.zeros(r,dtype=float)
    if licznik<=r<licznik**(2):
        warnings.warn("r is smaller than rank(rho)**(2), the output might be less precise")
    p=np.zeros(r,dtype=float)

    for i in range(min([r,n])):                         #Setting initial decomposition of rho
        p[i]=pqp[max([n-r,0])+i]
    sp=list(eig[1][max([n-r,0]):])                      #only eigenvectors with non-zero eigenvalues
    
    for i in range(r-min([r,n])):
        sp.append(qutip.Qobj(np.zeros(n)))
    for i in range(r):
        sp[i].dims=[dim,[1,1]]
    
    q=np.random.rand(r)
    q=q/sum(q)
    sq=[]
    sq1=[]
    sq2=[]
    for i in range(r):
            
                sta=random_haar_state(dim[0])
                stb=random_haar_state(dim[1])
                staaa=qutip.tensor(sta,stb)
                sq.append(staaa)                 #random separable decomposition 
                sq1.append(sta)
                sq2.append(stb)
    if qs is None:
        
        
        sq_mat = np.column_stack([s.full().ravel() for s in sq])  # shape: (d, n²)
    else:
        q=qs
        sq_mat=sqs
    sp_mat = np.column_stack([s.full().ravel() for s in sp])  # shape: (d, n²)

    while abs(Fp-F)>dif and itera<iteramax:
        itera+=1
        
    

        
              
        V = np.conj(sq_mat).T @ sp_mat
                
        V=V.T

        W = np.sqrt(np.outer(p, q))

        A = W * V
        
                            

        svdval=np.linalg.svd(A)
        
        u=np.conjugate(np.transpose(svdval[0]@svdval[2]))

        sqrt_p = np.sqrt(p)                         
        sp_weighted = sp_mat * sqrt_p[np.newaxis, :]       
        sp_weighted=sp_weighted.T

        
        

        alfa  = u @ sp_weighted

        pp=np.zeros(r,dtype=float)
        spp=[]
        for i in range(r):
            pp[i]=np.linalg.norm(alfa[i])**2
            spp.append(qutip.Qobj(alfa[i]/np.sqrt(pp[i])))   #updating decomposition of rho
            spp[i].dims=[dim,[1,1]]

#print(spp[len(sq)-1])
        
        for i in range(r):
                theta0=spp[i].ptrace(0)
                ccc=theta0.eigenstates(sort='high')
                cc=ccc[1][0]                                  #update separable decompsition
                theta1=spp[i].ptrace(1)
                ddd=theta1.eigenstates(sort='high')
                dd=ddd[1][0]
                sq1[i]=cc
                sq2[i]=dd
            

        qq=np.zeros(r,dtype=float)
        sss = 0
        overlaps = np.empty(r, dtype=float)
        sq111_mat = np.column_stack([s.full().ravel() for s in sq1]).T  # shape: (d, n²)
        sq222_mat = np.column_stack([s.full().ravel() for s in sq2]).T  # shape: (d, n²)
        sq_mat = np.einsum('ij,ik->ijk', sq111_mat, sq222_mat).reshape(r, n)


        spp_matri = np.column_stack([s.full().ravel() for s in spp]).T
        ov=np.conj(spp_matri) @ sq_mat.T
        overlaps=np.diag(ov)
        overlaps = abs(overlaps)**2
        
        sss=sum(pp*overlaps)


        

        
        qq=pp*overlaps/sss
        


        als_np = np.zeros((n, n), dtype=complex)
        als_np = np.einsum('i,ij,ik->jk', qq, sq_mat, np.conj(sq_mat))
        
        
        rhomm = rho.full()
        root_rho = scipy.linalg.sqrtm(rhomm)
        inner = root_rho @ als_np @ root_rho
        root_inner = scipy.linalg.sqrtm(inner)
        Fp=F

        F = np.real(np.trace(root_inner) ** 2)   #Computing fidelity
        
        sp_mat=copy.deepcopy(spp_matri.T)
        p=copy.deepcopy(pp)
        q=copy.deepcopy(qq)
        sq_mat=copy.deepcopy(sq_mat.T)

    #Delating small numerical inprecisions
    inner=qutip.Qobj(inner)
    inner.tidyup()
    root_inner = inner.sqrtm()
    F = np.real((root_inner.tr()) ** 2) 
    if dec:
        return [1-F,q,sq_mat]
    else:
        return 1-F






def fast_tensor_with_identity_at(ket_list, i, id_op):
    """
    Constructs the tensor product of the given list of Qobj kets,
    replacing the i-th ket (0-based index) with an identity operator.
    Fast version: assumes all kets are same-dimension Qobjs.
    
    Parameters:
    - ket_list: tuple or list of Qobj kets
    - i: index to replace with identity
    - id_op: precomputed identity operator (e.g., qeye(2))

    Returns:
    - Qobj: tensor product with identity at position i
    """
    n = len(ket_list)
    # Use tuple to avoid overhead
    ops = tuple(
        id_op if j == i else ket_list[j]
        for j in range(n)
    )
    return tensor(*ops)

def tensor_with_identity_at(ket_list, i):
    """
    Constructs the tensor product of the given list of Qobj kets,
    replacing the i-th ket (0-based index) with an identity operator.

    Parameters:
    - ket_list: list of Qobj kets (length n)
    - i: index to replace with identity (0 <= i < n)

    Returns:
    - Qobj: the resulting tensor product with identity at position i
    """
    ops = []
    for j, ket in enumerate(ket_list):
        if j == i:
            ops.append(qeye(ket.dims[0][0]))  # dimension of the ket
        else:
            ops.append(ket)
    return tensor(ops)

def uppersame(rho,dim,iteramax=2500,dif=10**(-8),r=None,qs=None,sqs=None,dec=False,sepitera=10):
    """
    Function which computes the upper bound of the geometric entanglement of the multipartite state rho. Dimensions of each subsystem must be equal.    
        Parameters
    ----------
    rho : array_like
        A 2D array representing a mixed state.

    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    

    iteramax : int, optional
        Maximum number of iterations of the algorithm. Default is 3000.

    dif : float, optional
        Parameter determining if the algorithm converged. Default is 10**(-7).

    r : int, optional
        Number of pure states in the separable decomposition of `rho`. 
        Default is `pro**2`, where `pro` are the subsystem dimensions of `rho`. 
        A recommended choice is `rank(rho)**2`.

    qs : list of float, optional
        Probabilities in the initial separable decomposition of `rho`.

    sqs : list of numpy.ndarray, optional
        Pure states (kets) in the initial separable decomposition of `rho`, 
        each given as a NumPy array.

    dec : bool, optional
        If True, the separable decomposition is also returned. 
        If False (default), only the bound is returned.
    
    sepitera : int, optional
        Number of iterations used to refine the separable decomposition.  
        This parameter controls the refinement depth of the separable 
        optimization loop. A typical value is between 10 and 20, although 
        the optimal number may depend strongly on the specific problem.  
        Lowering this value can speed up the computation but may reduce 
        the accuracy of the result.  
        Default is 10.
        
    Returns
    -------
    float or tuple
        If ``dec=False`` (default), returns the upper bound on the geometric 
        entanglement of `rho` as a float.  
        If ``dec=True``, returns a tuple ``(bound, decomposition)``, where 
        ``bound`` is the upper bound (float) and ``decomposition`` is the 
        separable decomposition (list of probabilities and pure states).


    Notes
    -----
    - These function computes upper bound using iteration alorithm. Each step of the algorithm finds a better
    separable decomposition of maximization variable.
    - Input state must be a multipartite state with subsystems having the same dimensions
    - Due to taking square roots and machine precision 10^(-16), accuracy of computation using upperbip is around 10^(-8). This accuracy is not
    included in the result, meaning that in principle upper bound computed by this function can be 10^(-8) lower then true value of the
    geometric entanglement.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> uppersame(rho,[2,2,2],2)  #Our variable is 2-symmetric extendible
    0.3025200226045961
    """
    
    #Checking correctness of inputs
    
    if len(dim)<=1:
        raise Exception("System should be at least a bipartite quantum state")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    rho=qutip.Qobj(rho)
    
    if abs(rho.tr()-1)>10**(-10):
        raise Exception("State should be normalized")
    if not rho.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = rho.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    producta=1
    

    for ih in dim:
        producta*=ih
    if len(rho.full())!=producta:
        raise Exception("Dimension of the system is not equal to the product of dimensions of subsystem")
    if len(set(dim)) != 1:
        raise Exception("All subsystems should have the same dimensions.")
    if len(dim)<=1:
        raise Exception("System must be a compound quantum system.")
    if r==None:
        r=producta**(2)

    if (not isinstance(iteramax,int)) or iteramax<=0:
        raise ValueError("iteramax should be an integer greater than 0.")
    if (not isinstance(sepitera,int)) or sepitera<=0:
        raise ValueError("sepitera should be an integer greater than 0.")
    if not isinstance(r,int):
        raise ValueError("r should be an integer")
    if (not isinstance(dif,float)) or not (0<dif<1):
        raise ValueError("dif should be in interval (0,1)")
    if (qs is None and (not sqs is None)) or (sqs is None and not (sqs is None)):
        raise Exception("Both or none of the qs, sqs must be None.")
    if not (sqs is None) and r!=len(sqs.T):
        raise Exception("sqs and r should have the same number of elements")
    if not (qs is None) and r!=len(qs):
        raise Exception("qs and r should have the same number of elements")
    
    if r>producta**(2):
        raise Exception(f"r should not be greater than dim(rho)**(2)={producta**(2)}")
    if not (dec==True or dec==False):
        raise Exception("dec should be True or False")

    
    #Main part of the function
    rho.dims=[dim,dim]
    F=69
    itera=0
    Fp=0
    n=len(rho.full())
    eig=rho.eigenstates()
    pqp=eig[0]
    pqp = np.where(pqp < 10**(-15), 0.0, pqp)               #Handling numerical errors in eigendecomposition
    licznik = np.count_nonzero(pqp > 10**(-15))
    
    pqp=pqp/sum(pqp)
    if r<licznik:
        raise Exception(f"Number r of vectors in decomposition is too small to represent rho. The number of non-zero eigenvalues of rho is {licznik}")
    p=np.zeros(r,dtype=float)
    if licznik<=r<licznik**(2):
        warnings.warn("r is smaller than rank(rho)**(2), the output might be less precise")
    p=np.zeros(r,dtype=float)
    
    
    for i in range(min([r,n])):                         #Setting initial decomposition of rho
        p[i]=pqp[max([n-r,0])+i]
    sp=list(eig[1][max([n-r,0]):])                      #only eigenvectors with non-zero eigenvalues
    
    for i in range(r-min([r,n])):
        sp.append(qutip.Qobj(np.zeros(n)))
    for i in range(r):
        sp[i].dims=[dim,[1,1]]
    sql=[]
    for i in range(r):
            sqo=[]
            for iu in range(len(dim)):                   #Setting initial separable decomposition
                sta=random_haar_state(dim[iu])          
                sqo.append(sta)
            sql.append(sqo)
    if qs is None:
        
        
        sq=[tensor(*kets) for kets in zip(sql)]
        q=np.random.rand(r)
        q=q/sum(q)
        sq_mat = np.column_stack([s.full().ravel() for s in sq])  # shape: (d, n²)
    else:
        q=qs
        sq_mat = sqs
        
    
    
    
    
    sp_mat = np.column_stack([s.full().ravel() for s in sp])  # shape: (d, n²)
    
    while abs(Fp-F)>dif and itera<iteramax:
        itera+=1
        
        

        
                #print(sp_mat)
        V = np.conj(sq_mat).T @ sp_mat
                #print(p,q)
        V=V.T
        
        W = np.sqrt(np.outer(p, q))
        
        A = W * V
        
                  

        svdval=np.linalg.svd(A)
        
        u=np.conjugate(np.transpose(svdval[0]@svdval[2]))

        sqrt_p = np.sqrt(p)                         # shape (n²,)
        sp_weighted = sp_mat * sqrt_p[np.newaxis, :]       # List of Qobj
        sp_weighted=sp_weighted.T

        
        

        alfa  = u @ sp_weighted

        pp=np.zeros(r,dtype=float)
        spp=[]
        for i in range(r):
            pp[i]=np.linalg.norm(alfa[i])**2
            spp.append(qutip.Qobj(alfa[i]/np.sqrt(pp[i])))       #updating decomposition of rho
            spp[i].dims=[dim,[1]*len(dim)]


        id_op=qutip.qeye(dim[0])
        for ins in range(sepitera):                #Number of iterations in each run. 
            tester=np.zeros(r)
            for i in range(r):
                
                theta=spp[i]
                pra=sql[i].copy()
                
                pre=qutip.tensor(*pra)
                pre.dims=[dim,[1]]
                for ni in range(len(dim)):
                    ten=sql[i]
                    
                    tt=fast_tensor_with_identity_at(ten, ni, id_op)
                    
                    sql[i][ni]=tt.dag()*theta
                    sql[i][ni]=sql[i][ni].unit()                              #update separable decompsition
                    sql[i][ni].dims=[[dim[ni]],[1]]
                ppo=qutip.tensor(*sql[i])
                if 1-abs(ppo.overlap(pre))<dif/1000:
                    tester[i]=1
            if np.all(tester==1):
                break
                    
               
        
        qq=np.zeros(r,dtype=float)
        sss = 0
        overlaps = np.empty(r, dtype=float)
        sq=[tensor(*kets) for kets in zip(sql)]
        sq_mat = np.column_stack([s.full().ravel() for s in sq]).T  # shape: (d, n²)



        spp_matri = np.column_stack([s.full().ravel() for s in spp]).T
        ov=np.conj(spp_matri) @ sq_mat.T
        overlaps=np.diag(ov)
        overlaps = abs(overlaps)**2
        
        sss=sum(pp*overlaps)


        

        
        qq=pp*overlaps/sss
        


        als_np = np.zeros((n, n), dtype=complex)
        als_np = np.einsum('i,ij,ik->jk', qq, sq_mat, np.conj(sq_mat))
        
        
        rhomm = rho.full()
        root_rho = scipy.linalg.sqrtm(rhomm)
        inner = root_rho @ als_np @ root_rho
        
        root_inner = scipy.linalg.sqrtm(inner)
        Fp=F

        F = np.real(np.trace(root_inner) ** 2)    #Computing fidelity
        
        sp_mat=copy.deepcopy(spp_matri.T)
        p=copy.deepcopy(pp)
        q=copy.deepcopy(qq)
        sq_mat=copy.deepcopy(sq_mat.T)

    #Delating small numerical inprecisions
    inner=qutip.Qobj(inner)
    inner.tidyup()
    root_inner = inner.sqrtm()
    F = np.real((root_inner.tr()) ** 2) 
    if dec:
        return [float(1-F),q,sq_mat]
    else:
        return float(1-F)


def uppermult(rho,dim,iteramax=2000,dif=10**(-7),r=None,qs=None,sqs=None,dec=False,sepitera=10):
    """
    Function which computes the upper bound of the geometric entanglement of the multipartite state rho.    
        Parameters
    ----------
    rho : array_like
        A 2D array representing a mixed state.

    dim : list of int
        Dimensions of each subsystem. Each entry should be a positive integer.
    

    iteramax : int, optional
        Maximum number of iterations of the algorithm. Default is 3000.

    dif : float, optional
        Parameter determining if the algorithm converged. Default is 10**(-7).

    r : int, optional
        Number of pure states in the separable decomposition of `rho`. 
        Default is `pro**2`, where `pro` are the subsystem dimensions of `rho`. 
        A recommended choice is `rank(rho)**2`.

    qs : list of float, optional
        Probabilities in the initial separable decomposition of `rho`.

    sqs : list of numpy.ndarray, optional
        Pure states (kets) in the initial separable decomposition of `rho`, 
        each given as a NumPy array.

    dec : bool, optional
        If True, the separable decomposition is also returned. 
        If False (default), only the bound is returned.
        
    sepitera : int, optional
        Number of iterations used to refine the separable decomposition.  
        This parameter controls the refinement depth of the separable 
        optimization loop. A typical value is between 10 and 20, although 
        the optimal number may depend strongly on the specific problem.  
        Lowering this value can speed up the computation but may reduce 
        the accuracy of the result.  
        Default is 10.
        
    Returns
    -------
    float or tuple
        If ``dec=False`` (default), returns the upper bound on the geometric 
        entanglement of `rho` as a float.  
        If ``dec=True``, returns a tuple ``(bound, decomposition)``, where 
        ``bound`` is the upper bound (float) and ``decomposition`` is the 
        separable decomposition (list of probabilities and pure states).

    

    Notes
    -----
    - These function computes upper bound using iteration alorithm. Each step of the algorithm finds a better
    separable decomposition of maximization variable.
    - Input state can be a general multipartite quantum state
    - Due to taking square roots and machine precision 10^(-16), accuracy of computation using upperbip is around 10^(-8). This accuracy is not
    included in the result, meaning that in principle upper bound computed by this function can be 10^(-8) lower then true value of the
    geometric entanglement.

    Examples
    --------
    >>> GHZ = np.sqrt(1/2)*(qutip.basis(8,0)+qutip.basis(8,7))
    >>> W = np.sqrt(1/3)*(qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4))
    >>> GHZ=GHZ.proj()
    >>> W=W.proj()
    >>> rho=0.6*GHZ+0.4*W
    >>> uppermult(rho,[2,2,2],2)  
    0.3025202112087888
    """
    rho=qutip.Qobj(rho)
    
    #Checking correctness of inputs
    if abs(rho.tr()-1)>10**(-10):
        raise Exception("State should be normalized")
    if len(dim)<=1:
        raise Exception("System should be at least a bipartite state")
    if min(dim)<=1:
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not all(isinstance(x,int) for x in dim):
        raise Exception("Dimension of subsystem must be an integer greater than 1")
    if not rho.isherm:
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    evals = rho.eigenenergies()
    if not np.all(evals >= -10**(-10)):
        raise Exception("State must be represented by a positive-semidefinite matrix.")
    producta=1
    for ih in dim:
        producta*=ih
    if len(rho.full())!=producta:
        raise Exception("Dimension of the system is not equal to the product of dimensions of subsystem")
    if len(dim)<=1:
        raise Exception("System must be a compound quantum system.")
    if r==None:
        r=producta**(2)



    if (not isinstance(iteramax,int)) or iteramax<=0:
        raise ValueError("iteramax should be an integer greater than 0.")
    if (not isinstance(sepitera,int)) or sepitera<=0:
        raise ValueError("sepitera should be an integer greater than 0.")
    if not isinstance(r,int):
        raise ValueError("r should be an integer")
    if (not isinstance(dif,float)) or not (0<dif<1):
        raise ValueError("dif should be in interval (0,1)")
    if (qs is None and (not sqs is None)) or (sqs is None and not (sqs is None)):
        raise Exception("Both or none of the qs, sqs must be None.")
    if not (sqs is None) and r!=len(sqs.T):
        raise Exception("sqs and r should have the same number of elements")
    if not (qs is None) and r!=len(qs):
        raise Exception("qs and r should have the same number of elements")
    
    if r>producta**(2):
        raise Exception(f"r should not be greater than dim(rho)**(2)={producta**(2)}")
    if not (dec==True or dec==False):
        raise Exception("dec should be True or False")
    
    
    #Main part of the function
    rho.dims=[dim,dim]
    F=69
    itera=0
    Fp=0
    n=len(rho.full())
    eig=rho.eigenstates()
    pqp=eig[0]
    pqp = np.where(pqp < 10**(-15), 0.0, pqp)
    licznik = np.count_nonzero(pqp > 10**(-15))                  #Handling numerical errors in eigendecomposition
    pqp=pqp/sum(pqp)
    if r<licznik:
        raise Exception(f"Number r of vectors in decomposition is too small to represent rho. The number of non-zero eigenvalues of rho is {licznik}")
    p=np.zeros(r,dtype=float)
    if licznik<=r<licznik**(2):
        warnings.warn("r is smaller than rank(rho)**(2), the output might be less precise")
    p=np.zeros(r,dtype=float)
    for i in range(min([r,n])):                         #Setting initial decomposition of rho
        p[i]=pqp[max([n-r,0])+i]
    sp=list(eig[1][max([n-r,0]):])                      #only eigenvectors with non-zero eigenvalues
    
    for i in range(r-min([r,n])):
        sp.append(qutip.Qobj(np.zeros(n)))
    for i in range(r):
        sp[i].dims=[dim,[1,1]]
    sql=[]          
    for i in range(r):                                 #Setting initial separable decomposition
            sqo=[]
            for iu in range(len(dim)):
                sta=random_haar_state(dim[iu])          
                sqo.append(sta)
            sql.append(sqo)       #We need sql later as an array
    if qs is None:
        
        
        sq=[tensor(*kets) for kets in zip(sql)]
        q=np.random.rand(r)
        q=q/sum(q)
        sq_mat = np.column_stack([s.full().ravel() for s in sq])  # shape: (d, n²)
    else:
        q=qs
        sq_mat = sqs
    
    
    sp_mat = np.column_stack([s.full().ravel() for s in sp])  # shape: (d, n²)
    
    while abs(Fp-F)>dif and itera<iteramax:
        itera+=1
        
        

        
                
        V = np.conj(sq_mat).T @ sp_mat
                
        V=V.T

        W = np.sqrt(np.outer(p, q))

        A = W * V
        
                            

        svdval=np.linalg.svd(A)
        
        u=np.conjugate(np.transpose(svdval[0]@svdval[2]))

        sqrt_p = np.sqrt(p)                         # shape (n²,)
        sp_weighted = sp_mat * sqrt_p[np.newaxis, :]       # List of Qobj
        sp_weighted=sp_weighted.T

        
        

        alfa  = u @ sp_weighted

        pp=np.zeros(r,dtype=float)
        spp=[]
        for i in range(r):
            pp[i]=np.linalg.norm(alfa[i])**2
            spp.append(qutip.Qobj(alfa[i]/np.sqrt(pp[i])))      #updating decomposition of rho
            spp[i].dims=[dim,[1]*len(dim)]

#print(spp[len(sq)-1])
        id_op=qutip.qeye(dim[0])
        for ins in range(sepitera):                   #Number of iterations in each run. 
            tester=np.zeros(r)
            for i in range(r):
                
                theta=spp[i]
                pra=sql[i].copy()
                
                pre=qutip.tensor(*pra)
                pre.dims=[dim,[1]]
                for ni in range(len(dim)):                              #update separable decompsition
                    ten=sql[i]
                    
                    tt=tensor_with_identity_at(ten, ni)
                    
                    sql[i][ni]=tt.dag()*theta
                    sql[i][ni]=sql[i][ni].unit()
                    sql[i][ni].dims=[[dim[ni]],[1]]
        
                ppo=qutip.tensor(*sql[i])
                if 1-abs(ppo.overlap(pre))<dif/1000:
                    tester[i]=1
            if np.all(tester==1):
                break
                      
               
        
        qq=np.zeros(r,dtype=float)
        sss = 0
        overlaps = np.empty(r, dtype=float)
        sq=[tensor(*kets) for kets in zip(sql)]
        sq_mat = np.column_stack([s.full().ravel() for s in sq]).T  # shape: (d, n²)



        spp_matri = np.column_stack([s.full().ravel() for s in spp]).T
        ov=np.conj(spp_matri) @ sq_mat.T
        overlaps=np.diag(ov)
        overlaps = abs(overlaps)**2
        
        sss=sum(pp*overlaps)


        

        
        qq=pp*overlaps/sss
        


        als_np = np.zeros((n, n), dtype=complex)
        als_np = np.einsum('i,ij,ik->jk', qq, sq_mat, np.conj(sq_mat))
        
        
        rhomm = rho.full()
        root_rho = scipy.linalg.sqrtm(rhomm)
        inner = root_rho @ als_np @ root_rho
        
        root_inner = scipy.linalg.sqrtm(inner)
        Fp=F

        F = np.real(np.trace(root_inner) ** 2)       #Computing fidelity
        
        sp_mat=copy.deepcopy(spp_matri.T)
        p=copy.deepcopy(pp)
        q=copy.deepcopy(qq)
        sq_mat=copy.deepcopy(sq_mat.T)

    #Delating small numerical inprecisions
    inner=qutip.Qobj(inner)
    inner.tidyup()
    root_inner = inner.sqrtm()
    F = np.real((root_inner.tr()) ** 2) 
    if dec:
        return [float(1-F),q,sq_mat]
    else:
        return float(1-F)



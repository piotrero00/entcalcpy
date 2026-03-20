import numpy as np
import entcalcpy as ent

def horodecki_ppt_state(a: float) -> np.ndarray:
    """
    Returns the Horodecki 3x3 PPT entangled state as a 9x9 NumPy array.
    
    Parameters:
        a (float): Parameter in [0, 1]
    
    Returns:
        rho (np.ndarray): 9x9 density matrix
    """
    if not (0 <= a <= 1):
        raise ValueError("Parameter 'a' must be in the interval [0, 1].")

    norm = 1 / (8 * a + 1)
    rho = np.zeros((9, 9), dtype=np.complex128)

    # Main diagonal
    diagonal = [a, a, a,
                a, a, a,
                (1 + a) / 2, a, (1 + a) / 2]
    for i in range(9):
        rho[i, i] = diagonal[i]

    # Off-diagonal elements
    rho[0, 4] = rho[0, 8] = rho[4, 0] = rho[8, 0] = a
    rho[4, 8] = rho[8, 4] = a

    off_diag = (np.sqrt(1 - a ** 2)) / 2
    rho[6, 8] = rho[8, 6] = off_diag

    return norm * rho

print("Execution of the whole script might take several minutes with MOSEK and around 20 minutes without MOSEK")

a=0.2
rho=horodecki_ppt_state(a)

l=ent.ge_mixed_sm(rho,[3,3])
print("Lower bound on geometric entanglement for {a} is:", l[0])


print("Attempting to calculate more accurate lower bound on the geometric entanglement using MOSEK...")
try:
    
    result = ent.ge_mixed_gr(rho,[3,3], solversdp="MOSEK") 
    print(f"Success! Result with MOSEK: {result[0]}")
    
except Exception as e:
    
    print("\n[NOTE] MOSEK solver is not installed or license is missing.")
    print("Falling back to the default free solver (e.g., SCS or ECOS).")
    print("For higher precision and speed, installing MOSEK is recommended.\n")
    
    result = ent.ge_mixed_gr(rho,[3,3]) 
    print(f"Result with default solver: {result[0]}")



u=ent.upperbip(rho,[3,3],dif=10**(-9),iteramax=5000)
print("Upper bound on geometric entaglement for {a} is", u )

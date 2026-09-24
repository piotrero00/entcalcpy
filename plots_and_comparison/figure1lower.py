#This script was created in entcalcpy version 0.1.3

import numpy as np
import entcalcpy as en

#This scripts generate data for Figure 1 lower bounds

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

result_bounds=[]

a=1/100
rho=horodecki_ppt_state(a)

for ai in range(101):
    a=ai/100
    
    rho=horodecki_ppt_state(a)
    
    lower=en.ge_mixed_gr(rho,[3,3],solversdp="MOSEK")
    result_bounds.append(lower)

with open ('figure1lower.txt', 'w') as filehandle:
    for lis in result_bounds:
        filehandle.write(f'{lis}\n')





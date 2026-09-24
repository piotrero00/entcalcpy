#This script was created in entcalcpy version 0.1.3


import numpy as np
import entcalcpy as en

# Reproducibility
SEED = 123456
np.random.seed(SEED)


# This script generates data for Figure 1 upper bounds

def horodecki_ppt_state(a: float) -> np.ndarray:
    """
    Returns the Horodecki 3x3 PPT entangled state as a 9x9 NumPy array.

    Parameters
    ----------
    a : float
        Parameter in [0, 1].

    Returns
    -------
    np.ndarray
        The 9x9 density matrix.
    """
    if not (0 <= a <= 1):
        raise ValueError("Parameter 'a' must be in the interval [0, 1].")

    norm = 1 / (8 * a + 1)
    rho = np.zeros((9, 9), dtype=np.complex128)

    diagonal = [
        a, a, a,
        a, a, a,
        (1 + a) / 2, a, (1 + a) / 2
    ]

    for i in range(9):
        rho[i, i] = diagonal[i]

    rho[0, 4] = rho[0, 8] = rho[4, 0] = rho[8, 0] = a
    rho[4, 8] = rho[8, 4] = a

    off_diag = np.sqrt(1 - a**2) / 2
    rho[6, 8] = rho[8, 6] = off_diag

    return norm * rho


result_bounds = []

# The first point uses a random initial separable decomposition.
a = 1 / 100
rho = horodecki_ppt_state(a)

upper = en.upperbip(
    rho,
    [3, 3],
    iteramax=5000,
    dif=1e-9,
    dec=True
)

result_bounds.append(upper[0])

# Each subsequent point starts from the decomposition found
# for the preceding value of a.
for ai in range(2, 101):
    a = ai / 100
    print(f"a = {a:.2f}")

    rho = horodecki_ppt_state(a)

    upper = en.upperbip(
        rho,
        [3, 3],
        iteramax=900,
        dif=1e-9,
        qs=upper[1],
        sqs=upper[2],
        dec=True
    )

    result_bounds.append(upper[0])
    print(f"upper bound = {upper[0]:.16e}")


with open("figure1upper.txt", "w", encoding="utf-8") as filehandle:
    for value in result_bounds:
        filehandle.write(f"{value:.17g}\n")

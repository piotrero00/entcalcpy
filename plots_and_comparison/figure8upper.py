#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np


# Lower bound for Figure 8
# Six-qubit hexagonal spin chain
# Subsystem B:D:F
# beta = 5

N = 6
J = -1
bet = 5


# Pauli operators acting on each of the 6 qubits
sx = []
sy = []
sz = []

for i in range(N):
    ops_x = [qutip.qeye(2) for _ in range(N)]
    ops_y = [qutip.qeye(2) for _ in range(N)]
    ops_z = [qutip.qeye(2) for _ in range(N)]

    ops_x[i] = qutip.sigmax()
    ops_y[i] = qutip.sigmay()
    ops_z[i] = qutip.sigmaz()

    sx.append(qutip.tensor(ops_x))
    sy.append(qutip.tensor(ops_y))
    sz.append(qutip.tensor(ops_z))


# Interaction part of the Hamiltonian
# Periodic chain:
# A-B, B-C, C-D, D-E, E-F, F-A

H_int = 0

for i in range(N):
    j = (i + 1) % N

    H_int += -J / 2 * (
        sx[i] * sx[j]
        + sy[i] * sy[j]
    )


# Magnetic-field part
H_field = sum(sz)


resultsp = []
hs = []
h=-2
H = H_int + h * H_field
HH = -bet * H
r = HH.expm()
state = r / r.tr()

state_BDF = state.ptrace([1, 3, 5])
upper=en.uppersame(
    state_BDF,
    [2,2,2],
    iteramax=5000,
    dif=1e-9,
    dec=True
)
resultsp.append(upper[0])
# step = 0.05
for hi in range(-195, 205, 5):

    h = hi / 100
    hs.append(h)

    H = H_int + h * H_field

    # Thermal state rho = exp(-beta H) / Z
    HH = -bet * H
    r = HH.expm()
    state = r / r.tr()

    # Qubits:
    # A = 0
    # B = 1
    # C = 2
    # D = 3
    # E = 4
    # F = 5
    #
    # We keep B:D:F
    state_BDF = state.ptrace([1, 3, 5])

    # Geometric entanglement of B:D:F
    state_BDF = state.ptrace([1, 3, 5])
    upper=en.uppersame(
        state_BDF,
        [2,2,2],
        qs=upper[1],
        sqs=upper[2],
        iteramax=500,
        dif=1e-9,
        dec=True
    )
    resultsp.append(upper[0])

    print("h =", h, "lower =", upper[0])


# Save results
with open("figure8upper_beta5a.txt", "w") as filehandle:
    for lis in resultsp:
        filehandle.write(f"{lis}\n")

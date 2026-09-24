#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np
import itertools


# ============================================================
# Figure 9 - lower bound
# Amplitude damping acting locally on GHZ and W
# q = 0, 0.01, ..., 1
# ============================================================


# ---------- GHZ and W states ----------

zero = qutip.basis(2, 0)
one = qutip.basis(2, 1)

GHZ = (
    qutip.tensor(zero, zero, zero)
    + qutip.tensor(one, one, one)
).unit()

W = (
    qutip.tensor(one, zero, zero)
    + qutip.tensor(zero, one, zero)
    + qutip.tensor(zero, zero, one)
).unit()

rho_GHZ = GHZ.proj()
rho_W = W.proj()


# ---------- Amplitude damping channel ----------

def amplitude_damping_three_qubits(rho, q):

    E1 = qutip.Qobj([
        [1, 0],
        [0, np.sqrt(1 - q)]
    ])

    E2 = qutip.Qobj([
        [0, np.sqrt(q)],
        [0, 0]
    ])

    kraus = [E1, E2]

    # Initial zero operator with correct dimensions
    rho_out = 0 * rho

    # Lambda_q x Lambda_q x Lambda_q
    for K1, K2, K3 in itertools.product(kraus, repeat=3):

        K = qutip.tensor(K1, K2, K3)

        rho_out += K * rho * K.dag()

    return rho_out


# ---------- Computation ----------

results_GHZ = []
results_W = []
q_values = []

for iq in range(101):

    q = iq / 100
    q_values.append(q)

    print(f"\nq = {q:.2f}")

    # Apply local amplitude damping
    state_GHZ = amplitude_damping_three_qubits(rho_GHZ, q)
    state_W = amplitude_damping_three_qubits(rho_W, q)

    # Lower bound - Eq. (19)
    lower_GHZ = en.ge_mixed_gr(
        state_GHZ,
        [2, 2, 2],
        sdpaccuracy="high",
        solversdp="MOSEK"
    )

    lower_W = en.ge_mixed_gr(
        state_W,
        [2, 2, 2],
        sdpaccuracy="high",
        solversdp="MOSEK"
    )

    results_GHZ.append(lower_GHZ[0])
    results_W.append(lower_W[0])

    print("GHZ =", lower_GHZ[0])
    print("W   =", lower_W[0])


# ---------- Save results ----------

with open("figure9lower_GHZ.txt", "w") as filehandle:
    for value in results_GHZ:
        filehandle.write(f"{value}\n")


with open("figure9lower_W.txt", "w") as filehandle:
    for value in results_W:
        filehandle.write(f"{value}\n")


with open("figure9_q.txt", "w") as filehandle:
    for q in q_values:
        filehandle.write(f"{q}\n")

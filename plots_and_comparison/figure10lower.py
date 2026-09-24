#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np
import itertools


# ============================================================
# Figure 10 - lower bound
# Local depolarizing channel acting on GHZ and W states
# p = 0, 0.01, ..., 0.50
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


# ---------- Depolarizing channel ----------

def depolarizing_three_qubits(rho, p):

    E0 = np.sqrt(1 - 3*p/4) * qutip.qeye(2)
    E1 = np.sqrt(p/4) * qutip.sigmax()
    E2 = np.sqrt(p/4) * qutip.sigmay()
    E3 = np.sqrt(p/4) * qutip.sigmaz()

    kraus = [E0, E1, E2, E3]

    rho_out = 0 * rho

    # Lambda_p x Lambda_p x Lambda_p
    for K1, K2, K3 in itertools.product(kraus, repeat=3):

        K = qutip.tensor(K1, K2, K3)

        rho_out += K * rho * K.dag()

    return rho_out


# ---------- Computation ----------

results_GHZ = []
results_W = []
p_values = []


# p = 0, 0.01, ..., 0.50
for ip in range(51):

    p = ip / 100
    p_values.append(p)

    print(f"\np = {p:.2f}")

    state_GHZ = depolarizing_three_qubits(rho_GHZ, p)
    state_W = depolarizing_three_qubits(rho_W, p)

    # Lower bound (19)
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

with open("figure10lower_GHZ.txt", "w") as filehandle:
    for value in results_GHZ:
        filehandle.write(f"{value}\n")


with open("figure10lower_W.txt", "w") as filehandle:
    for value in results_W:
        filehandle.write(f"{value}\n")


with open("figure10_p.txt", "w") as filehandle:
    for p in p_values:
        filehandle.write(f"{p}\n")

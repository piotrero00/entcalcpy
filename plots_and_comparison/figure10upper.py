#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np
import itertools


# ============================================================
# Figure 10 - upper bound with warm start
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

"""
# ---------- Computation ----------
# Separate random seeds and warm starts for GHZ and W,
# following figure9upper.py. Both series are enabled.

p_values = [ip / 100 for ip in range(51)]
results_GHZ = []
results_W = []

np.random.seed(123456)

state_GHZ = depolarizing_three_qubits(rho_GHZ, 0)
upper_GHZ = en.uppersame(
    state_GHZ, [2, 2, 2],
    iteramax=5000,
    dif=1e-9,
    dec=True
)
results_GHZ.append(upper_GHZ[0])
print("p = 0.00, GHZ =", upper_GHZ[0])

for p in p_values[1:]:
    state_GHZ = depolarizing_three_qubits(rho_GHZ, p)

    upper_GHZ = en.uppersame(
        state_GHZ, [2, 2, 2],
        iteramax=900,
        dif=1e-9,
        qs=upper_GHZ[1],
        sqs=upper_GHZ[2],
        sepitera=18,
        dec=True
    )

    results_GHZ.append(upper_GHZ[0])
    print(f"p = {p:.2f}, GHZ = {upper_GHZ[0]}")

# Save GHZ before starting the independent W computation.
with open("figure10upper_GHZ.txt", "w") as filehandle:
    for value in results_GHZ:
        filehandle.write(f"{value}\n")



np.random.seed(654321)

state_W = depolarizing_three_qubits(rho_W, 0)
upper_W = en.uppersame(
    state_W, [2, 2, 2],
    iteramax=5000,
    dif=1e-9,
    dec=True
)
results_W.append(upper_W[0])
print("p = 0.00, W =", upper_W[0])

for p in p_values[1:]:
    state_W = depolarizing_three_qubits(rho_W, p)

    upper_W = en.uppersame(
        state_W, [2, 2, 2],
        iteramax=500,
        dif=1e-9,
        qs=upper_W[1],
        sqs=upper_W[2],
        sepitera=12,
        dec=True
    )

    results_W.append(upper_W[0])
    print(f"p = {p:.2f}, W = {upper_W[0]}")

with open("figure10upper_W.txt", "w") as filehandle:
    for value in results_W:
        filehandle.write(f"{value}\n")

"""
print("Recomputing problematic states")
#Last terms for GHZ require recomputations to increase accuracy
np.random.seed(12345687)
for p in [0.45,0.46,0.47,0.48,0.49,0.50]:
    state_GHZ = depolarizing_three_qubits(rho_GHZ, p)

    upper_GHZ = en.uppersame(
        state_GHZ, [2, 2, 2],
        iteramax=5000,
        dif=1e-9,
        sepitera=12,
    )
    print(p)
    print(upper_GHZ)

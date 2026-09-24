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
np.random.seed(123456)
results_GHZ = []
results_W = []
q_values = []


q=0
state_GHZ = amplitude_damping_three_qubits(rho_GHZ, q)
upper_GHZ = en.uppersame(state_GHZ,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)

results_GHZ.append(upper_GHZ[0])

for iq in range(1,101):

    q = iq / 100
    

    print(f"\nq = {q:.2f}")

    # Apply local amplitude damping
    state_GHZ = amplitude_damping_three_qubits(rho_GHZ, q)
    

    # Lower bound - Eq. (19)
    upper_GHZ = en.uppersame(state_GHZ,[2,2,2],iteramax=900,
        dif=1e-9,
        qs=upper_GHZ[1],
        sqs=upper_GHZ[2],
        sepitera=18,
        dec=True)

    

    results_GHZ.append(upper_GHZ[0])
    

    print("GHZ =", upper_GHZ[0])
    

np.random.seed(654321)



q=0
state_W = amplitude_damping_three_qubits(rho_W, q)
upper_W = en.uppersame(state_W,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)

results_W.append(upper_W[0])

for iq in range(1,101):

    q = iq / 100
    

    print(f"\nq = {q:.2f}")

    # Apply local amplitude damping
    
    state_W = amplitude_damping_three_qubits(rho_W, q)

    # Lower bound - Eq. (19)
    upper_W = en.uppersame(state_W,[2,2,2],iteramax=500,
        dif=1e-9,
        qs=upper_W[1],
        sqs=upper_W[2],
        sepitera=12,
        dec=True)

    

    
    results_W.append(upper_W[0])

    print("W =", upper_W[0])

# ---------- Save results ----------

with open("figure9upper_GHZaa.txt", "w") as filehandle:
    for value in results_GHZ:
        filehandle.write(f"{value}\n")


with open("figure9upper_Wa.txt", "w") as filehandle:
    for value in results_W:
        filehandle.write(f"{value}\n")


import qutip
import entcalcpy as en

# ==========================================
# 1. State Preparation (Thermal State)
# ==========================================
# We construct a 3-qubit Heisenberg XX spin chain Hamiltonian
J = 1.0

sy1 = qutip.tensor(qutip.sigmay(), qutip.qeye(2), qutip.qeye(2))
sy2 = qutip.tensor(qutip.qeye(2), qutip.sigmay(), qutip.qeye(2))
sy3 = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.sigmay())

sx1 = qutip.tensor(qutip.sigmax(), qutip.qeye(2), qutip.qeye(2))
sx2 = qutip.tensor(qutip.qeye(2), qutip.sigmax(), qutip.qeye(2))
sx3 = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.sigmax())

# Hamiltonian for the XX model
H = -J/2 * (sy1*sy2 + sy2*sy3 + sy3*sy1 + sx1*sx2 + sx2*sx3 + sx3*sx1)

# Calculate the thermal state rho = exp(-beta * H) / Tr(exp(-beta * H))
beta = 1.5
r = (-beta * H).expm()
rho = r / r.tr()

# We evaluate a bipartite cut of the 3-qubit system: 2 qubits vs 1 qubit.
# Hence, the dimensions of the cut are 2^2 = 4 and 2^1 = 2.
dims_bip = [4, 2]

print("================================================================")
print("  entcalcpy: upperbip usage and parameter tuning                ")
print("================================================================")
print("This script demonstrates the usage of the upperbip function,")
print("which calculates the upper bound of the geometric entanglement")
print("using an iterative gradient-descent algorithm.")
print("Proper parameter usage strongly influences performance and accuracy.")
print("================================================================\n")

# ==========================================
# Examples 1 & 2: Random Initialization Effect
# ==========================================
print("--- Examples 1 & 2: Random Initialization ---")
print("The algorithm depends on an initial separable decomposition,")
print("which is generated randomly. For that reason, upperbip called")
print("with the exact same parameters can give slightly different results.")

res_try1 = en.upperbip(rho, dims_bip, iteramax=1000)
print(f"iteramax=1000 (first try) : {res_try1}")

res_try2 = en.upperbip(rho, dims_bip, iteramax=1000)
print(f"iteramax=1000 (second try): {res_try2}\n")

# ==========================================
# Example 3: The 'dif' Parameter
# ==========================================
print("--- Example 3: Tuning the 'dif' parameter (Threshold) ---")
print("We can control the accuracy and time of computations using 'dif' and 'iteramax'.")
print("In each step, the algorithm increases the fidelity between the input 'rho'")
print("and the separable decomposition. The algorithm stops when the increase")
print("in fidelity falls below the 'dif' threshold.")
print("Setting 'dif' too loose (e.g., 10^-5) might cause the algorithm to stop")
print("too early, before reaching the most accurate (tightest) upper bound.")

res_dif_loose = en.upperbip(rho, dims_bip, dif=10**(-5), iteramax=4500)
print(f"dif=10**(-5), iteramax=4500: {res_dif_loose}\n")

# ==========================================
# Example 4: The 'iteramax' Parameter
# ==========================================
print("--- Example 4: Tuning the 'iteramax' parameter ---")
print("Let's tighten the 'dif' threshold to 10^-9.")
print("'iteramax' sets the maximal number of iteration steps. If the number")
print("of iterations exceeds 'iteramax', the function stops computations.")
print("Setting 'iteramax' too low might cause an inaccurate (loose) result")
print("because the algorithm hasn't had enough steps to converge.")

res_dif_tight = en.upperbip(rho, dims_bip, dif=10**(-9), iteramax=4500)
print(f"dif=10**(-9), iteramax=4500: {res_dif_tight}\n")

print("NOTE: For advanced parameters 'qs' and 'sqs', please see the")
print("example script discussing multiple similar states for upper bounds.")

print("================================================================")
print("Test run completed.")

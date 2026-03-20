import qutip
import entcalcpy as en

# ==========================================
# 1. State Preparation (Thermal State)
# ==========================================
# We construct a 3-qubit Heisenberg XX spin chain Hamiltonian
J = 1.0

# Define Pauli matrices for each site
sy1 = qutip.tensor(qutip.sigmay(), qutip.qeye(2), qutip.qeye(2))
sy2 = qutip.tensor(qutip.qeye(2), qutip.sigmay(), qutip.qeye(2))
sy3 = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.sigmay())

sx1 = qutip.tensor(qutip.sigmax(), qutip.qeye(2), qutip.qeye(2))
sx2 = qutip.tensor(qutip.qeye(2), qutip.sigmax(), qutip.qeye(2))
sx3 = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.sigmax())

# We ignore sz as we are focusing on the XX model
# sz1 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2), qutip.qeye(2))
# sz2 = qutip.tensor(qutip.qeye(2), qutip.sigmaz(), qutip.qeye(2))
# sz3 = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.sigmaz())

# Hamiltonian for the XX model
H = -J/2 * (sy1*sy2 + sy2*sy3 + sy3*sy1 + sx1*sx2 + sx2*sx3 + sx3*sx1)

# Calculate the thermal state rho = exp(-beta * H) / Tr(exp(-beta * H))
beta = 1.5
r = (-beta * H).expm()
rho = r / r.tr()

# Subsystem dimensions for 3 qubits
dims = [2, 2, 2]

print("================================================================")
print("  entcalcpy: gekppt usage and parameter tuning                  ")
print("================================================================")
print("This script demonstrates how different parameters influence")
print("the performance of the gekppt function (k-symmetric extensions).")
print("We evaluate the lower bounds for a thermal state of a spin chain.")
print("================================================================\n")

# ==========================================
# Example 1: k=1 Extension
# ==========================================
print("--- Example 1: k=1 ---")
print("We find an accurate solution with k=1. However, the bound might")
print("be tighter.")
res_k1 = en.gekppt(rho, dims, 1)
print(f"Result (k=1): {res_k1}\n")

# ==========================================
# Example 2: k=2 Extension
# ==========================================
print("--- Example 2: k=2 ---")
print("We obtain a better (tighter) lower bound. However, it still")
print("might be improved further with higher values of k.")
res_k2 = en.gekppt(rho, dims, 2)
print(f"Result (k=2): {res_k2}\n")

# ==========================================
# Example 3: k=3 Extension (Using MOSEK)
# ==========================================
print("--- Example 3: k=3 (Using MOSEK) ---")
print("We recommend using MOSEK if optimization over large systems or")
print("higher k-extensions is being done. In this example, we obtain a")
print("series of lower bounds indexed by k, where the tightness of the")
print("bound increases with k.")

try:
    res_k3 = en.gekppt(rho, dims, 3, solversdp="MOSEK")
    print(f"Success! Result (k=3) with MOSEK: {res_k3}\n")
except Exception as e:
    print("\n[NOTE] MOSEK solver is not installed or the license is missing.")
    print("Falling back to default solver for k=3. This might take a moment...")
    res_k3_def = en.gekppt(rho, dims, 3)
    print(f"Result (k=3) with default solver: {res_k3_def}\n")

# ==========================================
# Example 4: k=4 Extension (Using MOSEK)
# ==========================================
print("--- Example 4: k=4 (Using MOSEK) ---")
print("MOSEK often finds solutions with higher than desired accuracy.")
print("You can modify MOSEK parameters inside the source code of entcalcpy,")
print("but if you do this, you should really know what you are doing.")

try:
    # We attempt k=4 with MOSEK.
    res_k4 = en.gekppt(rho, dims, 4, solversdp="MOSEK")
    print(f"Success! Result (k=4) with MOSEK: {res_k4}\n")
except Exception as e:
    print("\n[NOTE] MOSEK solver is not installed or the license is missing.")
    print("Skipping the k=4 example. Computing k=4 with free solvers")
    print("takes significantly longer. Install MOSEK to test this feature.\n")

print("================================================================")
print("Test run completed.")

import qutip
import entcalcpy as en

# ==========================================
# 1. State Preparation
# ==========================================
# Create a 3-qubit GHZ state projector and normalize
ghz = qutip.basis(8, 0) + qutip.basis(8, 7)
ghz = ghz.proj().unit()

# Create a 3-qubit W state projector and normalize
w = qutip.basis(8, 1) + qutip.basis(8, 2) + qutip.basis(8, 4)
w = w.proj().unit()

# Create a mixture: 70% GHZ and 30% W state
rho = 0.7 * ghz + 0.3 * w

# Subsystem dimensions for 3 qubits
dims = [2, 2, 2]

print("================================================================")
print("  entcalcpy: geppt usage and parameter tuning                   ")
print("================================================================")
print("This script demonstrates how different parameters influence")
print("the performance of the geppt function (PPT-based lower bound).")
print("Proper parameter usage is crucial here, as default solvers")
print("might struggle to find an accurate solution for this specific bound.")
print("================================================================\n")

# ==========================================
# Example 1: Default Parameters (Inaccurate Result)
# ==========================================
print("--- Example 1: Default values ---")
print("Calculating with default parameters...")
print("We often find an inaccurate solution with default settings for geppt.")
print("The solver (SCS) might finish, but flag the result as inaccurate.")
res_default = en.geppt(rho, dims)
print(f"Result: {res_default}\n")

# ==========================================
# Example 2: Increasing Iterations
# ==========================================
print("--- Example 2: Increasing iterations (itera=100000) ---")
print("We need to increase the iteration number. Increasing the number")
print("of iterations sometimes helps. Let's try 100,000 iterations.")
res_itera = en.geppt(rho, dims, itera=100000)
print(f"Result: {res_itera}")
print("As we can see, for this specific problem, the result is still")
print("flagged as inaccurate. In such cases, we highly recommend")
print("switching the solver to MOSEK.\n")

# ==========================================
# Example 3: Using the MOSEK Solver
# ==========================================
print("--- Example 3: Using MOSEK solver ---")
print("Using MOSEK as a solver can decrease the required number of iterations")
print("and significantly speed up computations. More importantly, MOSEK often")
print("finds an accurate solution where SCS has numerical problems.")

try:
    # We attempt to use MOSEK.
    res_mosek = en.geppt(rho, dims, solversdp="MOSEK")
    print(f"Success! Result with MOSEK: {res_mosek}\n")
except Exception as e:
    print("\n[NOTE] MOSEK solver is not installed or the license is missing.")
    print("Skipping the MOSEK example. See documentation for installation instructions.\n")

# ==========================================
# Example 4: Modifying MOSEK Accuracy
# ==========================================
print("--- Example 4: MOSEK with modified accuracy (sdpaccuracy=10^-5) ---")
print("MOSEK often finds solutions with higher than the explicitly desired accuracy.")
print("You can modify MOSEK parameters directly inside the entcalcpy source code,")
print("but if you do this, you should really know what you are doing.")

try:
    res_mosek_acc = en.geppt(rho, dims, solversdp="MOSEK", sdpaccuracy=10**(-5))
    print(f"Success! Result with MOSEK (lower accuracy target): {res_mosek_acc}\n")
except Exception as e:
    print("\n[NOTE] MOSEK solver is not installed or the license is missing.")
    print("Skipping this example.\n")

print("================================================================")
print("Test run completed.")

import qutip
import entcalcpy as en

# ==========================================
# 1. State Preparation (Ensemble)
# ==========================================
# Create a 3-qubit GHZ state vector (ket) and normalize
ghz = qutip.basis(8, 0) + qutip.basis(8, 7)
ghz = ghz.unit()

# Create a 3-qubit W state vector (ket) and normalize
w = qutip.basis(8, 1) + qutip.basis(8, 2) + qutip.basis(8, 4)
w = w.unit()

# Define the ensemble: probabilities and corresponding pure states
states = [ghz, w]
probs = [0.7, 0.3]

# Subsystem dimensions for 3 qubits
dims = [2, 2, 2]

print("================================================================")
print("  entcalcpy: ge_mixed_ra_sm usage and parameter tuning          ")
print("================================================================")
print("This script demonstrates how different parameters influence")
print("the performance and accuracy of the ge_mixed_ra_sm function.")
print("Like ge_mixed_ra_gr, it takes probabilities and corresponding")
print("pure states as input. Thanks to this, it is able to find")
print("a more efficient purification.")
print("================================================================\n")

# ==========================================
# Example 1: Default Parameters
# ==========================================
print("--- Example 1: Default values ---")
print("Calculating with default parameters...")
print("We usually find an accurate solution with these settings.")
res_default = en.ge_mixed_ra_sm(states, probs, dims)
print(f"Result: {res_default}\n")

# ==========================================
# Example 2: Limiting Iterations (Inaccurate Result)
# ==========================================
print("--- Example 2: Too few iterations (itera=100) ---")
print("Setting 'itera=100' might not be enough for the default solver to converge.")
print("This causes the result to be artificially higher than the true lower bound")
print("and is indicated by an 'inaccurate' string in the output tuple.")
res_itera = en.ge_mixed_ra_sm(states, probs, dims, itera=100)
print(f"Result: {res_itera}\n")

# ==========================================
# Example 3: Using the MOSEK Solver
# ==========================================
print("--- Example 3: Using MOSEK solver ---")
print("Using MOSEK as a solver can significantly decrease the required number")
print("of iterations and speed up computations. If you cannot obtain an accurate")
print("result in a reasonable number of iterations using the default (SCS) solver,")
print("we highly recommend switching to MOSEK.")

try:
    # We attempt to use MOSEK. If the user doesn't have it installed/licensed,
    # the script will gracefully catch the error.
    res_mosek = en.ge_mixed_ra_sm(states, probs, dims, solversdp="MOSEK")
    print(f"Success! Result with MOSEK: {res_mosek}\n")
except Exception as e:
    print("\n[NOTE] MOSEK solver is not installed or the license is missing.")
    print("Skipping the MOSEK example. See documentation for installation instructions.\n")

# ==========================================
# Example 4: Decreasing Required Accuracy
# ==========================================
print("--- Example 4: Decreasing accuracy (sdpaccuracy=10^-5, itera=100) ---")
print("One can also decrease the desired accuracy. The true lower bound lies")
print("within the 'sdpaccuracy' range from the returned result.")
print("If you need to quickly estimate the bound, decreasing the required accuracy")
print("is a good option. We do not recommend putting too strict accuracy (like 10^-10),")
print("since the upper bound accuracy is natively around 10^-8 due to machine precision.")
res_acc = en.ge_mixed_ra_sm(states, probs, dims, itera=100, sdpaccuracy=10**(-5))
print(f"Result: {res_acc}\n")

print("================================================================")
print("Test run completed.")

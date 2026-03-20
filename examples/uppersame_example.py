import qutip
import numpy as np
import entcalcpy as en

print("================================================================")
print("  entcalcpy: uppersame usage and accuracy discussion            ")
print("================================================================")
print("This script demonstrates the usage of the uppersame function.")
print("Unlike uppermult, uppersame only accepts systems composed of")
print("subsystems with IDENTICAL dimensions (e.g., [2, 2, 2]).")
print("Thanks to this assumption, it is significantly faster.")
print("")
print("[WARNING] Testing parameters on a fully random state requires")
print("more iterations to converge. This script might take a few")
print("minutes to complete.")
print("================================================================\n")

# ==========================================
# 1. State Preparation (GHZ Pure State)
# ==========================================
# Create a 3-qubit GHZ state projector and normalize
ghz = qutip.basis(8, 0) + qutip.basis(8, 7)
ghz = ghz.unit()
ghz = ghz.proj()

# ==========================================
# Example 1: Numerical Precision (Machine Epsilon)
# ==========================================
print("--- Example 1: Numerical Precision and Machine Epsilon ---")
print("We run uppersame 20 times on the exact same GHZ state.")
print("Calculating...")

results = []
for i in range(20):
    res = en.uppersame(ghz, [2, 2, 2])
    results.append(res)

print(f"Minimum result: {min(results)}")
print(f"Maximum result: {max(results)}\n")

print("It is known that the geometric entanglement of the GHZ state is 0.5.")
print("We obtain results that might be around 10^-9 greater OR lower.")
print("Results slightly lower than 0.5 might be surprising, because these")
print("functions should output an UPPER bound. This happens due to machine")
print("precision. Since we are taking square roots of machine precision (10^-16),")
print("we might encounter a ~10^-8 error in the results. One needs to keep")
print("in mind that the strict upper bound can be this amount higher than the output.\n")

# ==========================================
# Example 2: Extracting Separable Decomposition
# ==========================================
print("--- Example 2: Verifying the Decomposition (dec=True) ---")
print("If you have doubts about the result, you can check the correctness")
print("of the separable decomposition using the optional argument dec=True.")
res_dec = en.uppersame(ghz, [2, 2, 2], dec=True)

print(f"Upper bound: {res_dec[0]}")
print(f"Probability distribution: {res_dec[1]}")
print(f"Kets in each column:\n{res_dec[2]}\n")

# ==========================================
# 2. State Preparation (GHZ-W Mixture)
# ==========================================
# Create a 3-qubit W state projector and normalize
w = qutip.basis(8, 1) + qutip.basis(8, 2) + qutip.basis(8, 4)
w = w.unit()
w = w.proj()

print("================================================================")
print("  Parameter Tuning Examples                                     ")
print("================================================================")

# Example 3: Default values on a simple state
print("\n--- Example 3: Default values (GHZ-W mixture) ---")
rho = 0.6 * ghz + 0.4 * w
b = en.uppersame(rho, [2, 2, 2])
print(f"Default values result for simple rank-2 state: {b}\n")

# ==========================================
# 3. Complex State for Parameter Tuning
# ==========================================
print("--- Testing parameters on a complex state ---")
print("To truly see the impact of parameters like 'iteramax', 'dif', and 'r',")
print("we generate a completely random, full-rank 3-qubit density matrix.")
rho_hard = qutip.rand_dm(8)
dims_3q = [2, 2, 2]

# ==========================================
# Example 4: The 'r' Parameter
# ==========================================
print("\n--- Example 4: The 'r' Parameter (r=4) ---")
print("By Caratheodory's theorem, the optimal separable decomposition has r <= d^2.")
print("We accelerate computations by taking a smaller r (e.g., r=4). Keep in mind")
print("that we might decrease the accuracy by doing this.")
res_r = en.uppersame(rho_hard, dims_3q, r=10)
print(f"Result (r=10): {res_r}\n")

# ==========================================
# Example 5: The 'sepitera' Parameter
# ==========================================
print("--- Example 5: The 'sepitera' Parameter (sepitera=25) ---")
print("By increasing 'sepitera', we can increase the number of outer iterations")
print("before convergence. But be careful: setting 'sepitera' too high slows")
print("down computations without significantly increasing accuracy.")
res_sep = en.uppersame(rho_hard, dims_3q, r=10, sepitera=25)
print(f"Result (sepitera=25, r=10): {res_sep}\n")

# ==========================================
# Example 6: The 'dif' Parameter
# ==========================================
print("--- Example 6: The 'dif' Parameter (dif=10^-5) ---")
print("A larger 'dif' makes computations faster, but the result might not be")
print("perfectly converged. The function terminates if the fidelity update")
print("between subsequent iterations is smaller than 'dif'.")
res_dif = en.uppersame(rho_hard, dims_3q, r=10, dif=10**(-5))
print(f"Result (dif=10^-5, r=10): {res_dif}\n")

# ==========================================
# Example 7: The 'iteramax' Parameter
# ==========================================
print("--- Example 7: The 'iteramax' Parameter (iteramax=10 vs high) ---")
print("'iteramax' determines the maximal number of iterations. It prevents")
print("endlessly long computations.")
res_itera_low = en.uppersame(rho_hard, dims_3q, r=10, iteramax=10)
print(f"Result (iteramax=10 - very fast, inaccurate): {res_itera_low}")

res_itera_high = en.uppersame(rho_hard, dims_3q, r=10, iteramax=1500)
print(f"Result (iteramax=1500 - slower, highly accurate): {res_itera_high}\n")

print("================================================================")
print("Test run completed.")

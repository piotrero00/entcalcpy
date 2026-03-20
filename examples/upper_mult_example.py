import qutip
import numpy as np
import entcalcpy as en

print("================================================================")
print("  entcalcpy: uppermult usage and accuracy discussion            ")
print("================================================================")
print("This script demonstrates the usage of the uppermult function")
print("with optional parameters. We also discuss the numerical accuracy")
print("of computations and how to extract the separable decomposition.")
print("")
print("[WARNING] Testing parameters on a fully random state requires")
print("more iterations to converge. This script might take around")
print("10 minutes to complete.")
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
print("We run uppersame/uppermult 20 times on the exact same GHZ state.")
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
res_dec = en.uppermult(ghz, [2, 2, 2], dec=True)

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

# Create a mixture: 60% GHZ and 40% W state
rho = 0.6 * ghz + 0.4 * w
dims_3q = [2, 2, 2]

print("================================================================")
print("  Parameter Tuning for Complex States                           ")
print("================================================================")

# ==========================================
# Example 3: Complex State for Parameter Tuning
# ==========================================
print("\n--- Testing parameters on a complex state ---")
print("The GHZ-W mixture above is a simple, rank-2 state. The algorithm finds")
print("the global optimum so easily that parameter tweaking doesn't change the result.")
print("To truly see the impact of parameters like 'iteramax', 'dif', and 'r',")
print("we now generate a completely random, full-rank 3-qubit density matrix.")

# Generate a random 8x8 density matrix (full rank = 8)
rho_hard = qutip.rand_dm(8)

# Now use rho_hard for the parameter examples:
res_fast = en.uppermult(rho_hard, [2, 2, 2], iteramax=100)
print(f"Result with extremely low iteramax (iteramax=100): {res_fast}")

res_accurate = en.uppermult(rho_hard, [2, 2, 2], iteramax=1000)
print(f"Result with greater iteramax (iteramax=1000): {res_accurate}\n")

# ==========================================
# Example 4: The 'r' Parameter (Rank / Caratheodory's theorem)
# ==========================================
print("--- Example 4: The 'r' Parameter (r=10) ---")
print("By Caratheodory's theorem, we are guaranteed that the optimal separable")
print("decomposition has r <= d^2, where d is the dimension of rho.")
print("We can accelerate computations by taking a smaller r (e.g., r=10 for 3 qubits),")
print("but keep in mind that we might decrease the accuracy by doing so.")
res_r = en.uppermult(rho_hard, dims_3q, r=10)
print(f"Result (r=10): {res_r}\n")

# ==========================================
# Example 5: The 'sepitera' Parameter
# ==========================================
print("--- Example 5: The 'sepitera' Parameter (sepitera=25, r=4) ---")
print("By increasing 'sepitera', we can increase the number of outer iterations")
print("before convergence. But be careful: setting 'sepitera' too high slows")
print("down computations without significantly increasing accuracy.")
res_sep = en.uppermult(rho_hard, dims_3q, r=10, sepitera=25)
print(f"Result (sepitera=25, r=10): {res_sep}\n")

# ==========================================
# Example 6: The 'dif' Parameter (Threshold)
# ==========================================
print("--- Example 6: The 'dif' Parameter (dif=10^-5, r=4) ---")
print("A larger 'dif' makes computations faster, but the result might not be")
print("perfectly converged. The function terminates if the fidelity update")
print("between subsequent iterations is smaller than 'dif'.")
res_dif = en.uppermult(rho_hard, dims_3q, r=10, dif=10**(-5))
print(f"Result (dif=10^-5, r=10): {res_dif}\n")

# ==========================================
# Example 7: The 'iteramax' Parameter
# ==========================================
print("--- Example 7: The 'iteramax' Parameter (iteramax=1500) ---")
print("'iteramax' determines the maximal number of iterations. It prevents")
print("endlessly long computations. Note that with dec=False, the output")
print("is a float number, while with dec=True, it is a tuple.")
res_itera = en.uppermult(rho_hard, dims_3q, iteramax=1500)
print(f"Result (iteramax=1500): {res_itera}\n")

# ==========================================
# Example 8: Bipartite Cuts with uppermult
# ==========================================
print("--- Example 8: Different Dimensions / Bipartite Cuts ---")
print("uppermult is also able to compute the upper bound for systems with")
print("different dimensions (e.g., a bipartite cut dims=[4, 2]).")
print("NOTE: For bipartite systems, upperbip is usually more efficient.")
print("For multipartite systems, uppermult is the recommended choice.")
# Here we can safely use the original rho for a quick demonstration of API usage
res_bip = en.uppermult(rho, [4, 2], r=10, iteramax=500)
print(f"Result (dims=[4,2], iteramax=500): {res_bip}\n")

print("================================================================")
print("Test run completed.")

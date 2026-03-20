import qutip
import numpy as np
import entcalcpy as en

print("================================================================")
print("  entcalcpy: ge_pure usage and accuracy discussion              ")
print("================================================================")
print("This script demonstrates the usage of the ge_pure function.")
print("It calculates the geometric entanglement for pure states,")
print("returning both the lower bound and the estimation error caused")
print("by relaxing the separability condition to the PPT condition.")
print("")
print("NOTE: The first value is a rigorous lower bound, not just an")
print("estimate. The true value of the geometric entanglement lies")
print("exactly at the distance 'error' from the lower bound (up to")
print("the numerical accuracy of the underlying SDP solver).")
print("================================================================\n")

# ==========================================
# Example 1: 3-Qubit GHZ State
# ==========================================
print("--- Example 1: 3-Qubit GHZ State ---")
ghz = qutip.basis(8, 0) + qutip.basis(8, 7)
ghz = ghz.unit()
dims_3q = [2, 2, 2]

print("Calculating...")
res_3q = en.ge_pure(ghz, dims_3q)
print(f"Result (Lower bound, Error): {res_3q}")

print("Notice that the second value in the output (the error) is exactly 0.")
print("This means the lower bound is strictly equal to the true value.")
print("This is because the optimization involves a bipartite cut where")
print("one subsystem is a qubit. For qubit-qubit or qubit-qutrit systems,")
print("the set of separable states is strictly equal to the set of PPT states.\n")

# ==========================================
# Example 2: 5-Qubit GHZ State
# ==========================================
print("--- Example 2: 5-Qubit GHZ State ---")
ghz5q = qutip.basis(32, 0) + qutip.basis(32, 31)
ghz5q = ghz5q.unit()
dims_5q = [2, 2, 2, 2, 2]

print("Calculating for 5 qubits...")
res_5q = en.ge_pure(ghz5q, dims_5q)
print(f"Result (Lower bound, Error): {res_5q}")

print("For higher dimensions, there is generally a non-zero estimation error")
print("caused by the geometric difference between the separable and PPT sets.")
print("However, for specific states (like the GHZ state), the error can be exactly 0.")
print("This happens when the closest PPT state found by the algorithm is diagonal,")
print("since diagonal states are always completely separable.\n")

# ==========================================
# Example 3: 5-Qubit Random Pure States
# ==========================================
print("--- Example 3: 5-Qubit Random Pure State ---")
print("Calculating for 5 completely random 5-qubit states...")

for i in range(5):
    x = qutip.rand_ket(32)
    res_rand = en.ge_pure(x, dims_5q)
    print(f"Result (Lower bound, Error): {res_rand}")

print("\nThis demonstrates that the function can efficiently handle general")
print("multipartite pure states without relying on any special symmetries.")
print("Notice how the true geometric entanglement value always lies at the exact")
print("distance 'error' from the lower bound, maintaining consistent precision.")

print("\n================================================================")
print("Test run completed.")

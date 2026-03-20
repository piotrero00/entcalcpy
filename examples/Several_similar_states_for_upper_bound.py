import qutip
import numpy as np
import entcalcpy as en
import time

print("================================================================")
print("  entcalcpy: uppersame and 'warm start' optimization            ")
print("================================================================")
print("This script demonstrates an advanced usage of the uppersame")
print("function (and upperbip). It compares a naive computation approach")
print("with an accelerated 'warm start' approach for similar states.")
print("By passing the decomposition components (qs, sqs) from a previous")
print("state, we can drastically reduce the required number of iterations.")
print("")
print("[WARNING] Execution of the whole script on a standard laptop")
print("lasts approximately 25 minutes. Please be patient.")
print("================================================================\n")

# ==========================================
# 1. Operator Setup (Global)
# ==========================================
print("--- 1. Initializing Hamiltonian and Thermal States ---")
def get_hamiltonian(J=1.0):
    """Generates the Hamiltonian for a 3-spin chain."""
    # Create operators using list comprehension for cleaner code
    sx = [qutip.tensor([qutip.sigmax() if j == i else qutip.qeye(2) for j in range(3)]) for i in range(3)]
    sy = [qutip.tensor([qutip.sigmay() if j == i else qutip.qeye(2) for j in range(3)]) for i in range(3)]
    
    # H = -J/2 * sum(S_i * S_{i+1}) with periodic boundary conditions
    h_int = 0
    for i in range(3):
        next_i = (i + 1) % 3
        h_int += (sy[i] * sy[next_i] + sx[i] * sx[next_i])
    
    return -J / 2 * h_int

H_base = get_hamiltonian() # We create the Hamiltonian of the XX model.

def get_thermal_state(H, beta):
    """Computes the normalized thermal state: rho = exp(-beta*H) / Tr(exp(-beta*H))."""
    rho = (-H * beta).expm()
    return rho / rho.tr()

print("Hamiltonian constructed. Starting computations...\n")

# ==========================================
# 2. Naive Approach: Random Initialization
# ==========================================
print("--- 2. Starting Naive Computations (Random Start) ---")
print("Computing upper bounds for varying temperatures (beta from 0.60 to 0.95).")
print("Starting from a random decomposition every time requires a high iteramax (4000).")

start_naive = time.time()
results_naive = []

for i in range(60, 100, 5):
    beta = i / 100
    state = get_thermal_state(H_base, beta)
    
    # Naive call: random start, high iteration limit
    res = en.uppersame(state, [2, 2, 2], iteramax=4000, dif=10**-8)
    results_naive.append(res)
    print(f"Beta: {beta:.2f} | Upper bound: {res:.9f}")

end_naive = time.time()
print(f"-> Naive approach took: {end_naive - start_naive:.2f} seconds.\n")

# ==========================================
# 3. Accelerated Approach: Warm Start
# ==========================================
print("--- 3. Starting Accelerated Computations (Warm Start) ---")
print("We use the decomposition (qs, sqs) of the previous state as the initial")
print("seed for the next state. This allows us to slash iteramax from 4000 to 500!")

start_acc = time.time()
results_acc = []

# Initial reference state
beta_init = 0.60
state = get_thermal_state(H_base, beta_init)

# First computation: must start from a random decomposition to get initial qs/sqs.
# Note: dec=True is required to return the decomposition components in the tuple.
print(f"Computing initial seed for Beta: {beta_init:.2f}...")
res = en.uppersame(state, [2, 2, 2], iteramax=4000, dif=10**-9, dec=True)
results_acc.append(res[0])
print(f"Beta: {beta_init:.2f} | Upper bound (seed) : {res[0]:.9f}")

# Iterative loop using previous results as seeds
for i in range(65, 100, 5):
    beta = i / 100
    state = get_thermal_state(H_base, beta)
    
    # Pass qs (probabilities) and sqs (state ensemble) from the previous step.
    # We use res[1] and res[2] because dec=True makes the function return a tuple.
    res = en.uppersame(
        state, [2, 2, 2], 
        iteramax=500,  # Drastically reduced!
        dif=10**-8, 
        dec=True, 
        qs=res[1], 
        sqs=res[2]
    )
    
    results_acc.append(res[0])
    print(f"Beta: {beta:.2f} | Upper bound (accel): {res[0]:.9f}")

end_acc = time.time()
print(f"-> Accelerated approach took: {end_acc - start_acc:.2f} seconds.")

print("\n================================================================")
print("Test run completed. Notice the massive difference in execution time!")

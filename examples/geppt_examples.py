import qutip
import entcalcpy as en


ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.proj()
ghz=ghz.unit()

w=qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4)
w=w.proj()
w=w.unit()

rho=0.7*ghz+0.3*w

"""
We demostrate different usage of parameters for geppt.
Proper parametr usage influences performance of the functions.
"""

print("Default values:",en.geppt(rho,[2,2,2]))
"""
We found inaccurate solution. We need to increase iteration number.
Increasing number of iterations sometimes helps
"""
print("itera=100000:",en.geppt(rho,[2,2,2],itera=100000))
"""
We obtained result, which is still inaccurate.
In such case we recommend switching solver to MOSEK.
"""

print("solversdp='MOSEK'",en.geppt(rho,[2,2,2],solversdp="MOSEK"))
"""
Using MOSEK as a solver can decrease required number of iterations and speed up significantly computations.
If you cannot obtain accurate result in reasonable number of iteration using SCS (default) solver,
we recommend switching to MOSEK. MOSEK often can finds solution where SCS has problems.
"""
print("solversdp='MOSEK',sdpaccuracy=10**(-5):",en.geppt(rho,[2,2,2],solversdp="MOSEK",sdpaccuracy=10**(-5)))
"""
MOSEK often find solutions with higher then desired accuracy.
You can modify MOSEK parameters, inside the code. But if you
do this, you should really know what you are doing.
"""

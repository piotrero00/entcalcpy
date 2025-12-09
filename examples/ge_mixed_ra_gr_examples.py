import qutip
import entcalcpy as en


ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.unit()

w=qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4)
w=w.unit()

rho=0.7*ghz+0.3*w

"""
We demostrate different usage of parameters for ge_mixed_ra_gr.
Proper parametr usage influences performance of the functions.
This function takes probabilities and corresponding pure

"""

print("Default values:",en.ge_mixed_ra_gr([ghz,w],[0.7,0.3],[2,2,2]))
"""
We found accurate solution. However, if we set too little number of iterations
we would not found a proper lower bound.
"""
print("itera=100",en.ge_mixed_ra_gr([ghz,w],[0.7,0.3],[2,2,2],itera=100))
"""
We obtained result, which is around 4*10**(-7) higher than lower bound computed eariler.
Setting too little iteration causes that we obtain result higher than lower bound. It is indicated by innacurate
string in the second position in the result.

"""

print("solversdp='MOSEK'",en.ge_mixed_ra_gr([ghz,w],[0.7,0.3],[2,2,2],solversdp="MOSEK"))
"""
Using MOSEK as a solver can decrease required number of iterations and speed up significantly computations.
If you cannot obtain accurate result in reasonable number of iteration using SCS (default) solver,
we recommend switching to MOSEK
"""
print("sdpaccuracy=10**(-5),itera=100:",en.ge_mixed_ra_gr([ghz,w],[0.7,0.3],[2,2,2],itera=100,sdpaccuracy=10**(-5)))
"""
One can also decrease desired accuracy. True lower bound lies within sdpaccuracy range from the result
of ge_mixed_gr. If one needs to quickly estimate the bound, decreasing required accuracy is a good option. We do not recommend
putting to strict accuracy, like 10**(-10), since upper bound accuracy is around 10**(-8). 10**(-8) is also default accuracy
in the function described in this script
"""

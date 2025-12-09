import qutip
import entcalcpy as en


J=1.

sy1=qutip.tensor(qutip.sigmay(),qutip.qeye(2),qutip.qeye(2))
sy2=qutip.tensor(qutip.qeye(2),qutip.sigmay(),qutip.qeye(2))
sy3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmay())

sx1=qutip.tensor(qutip.sigmax(),qutip.qeye(2),qutip.qeye(2))
sx2=qutip.tensor(qutip.qeye(2),qutip.sigmax(),qutip.qeye(2))
sx3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmax())

sz1=qutip.tensor(qutip.sigmaz(),qutip.qeye(2),qutip.qeye(2))
sz2=qutip.tensor(qutip.qeye(2),qutip.sigmaz(),qutip.qeye(2))
sz3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmaz())


bet=1.5
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)

HH=-H*bet
r=HH.expm()
rho=r/r.tr()



"""
We demostrate different usage of parameters for gekppt.
Proper parametr usage influences performance of the functions.
"""

print("k=1:",en.gekppt(rho,[2,2,2],1))
"""
We found accurate solution with k=1. However, the bound might
be tighter.
"""
print("k=2:",en.gekppt(rho,[2,2,2],2))
"""
We obtained better lower bound. However, it still might be
better.
"""

print("k=3:",en.gekppt(rho,[2,2,2],3,solversdp="MOSEK"))
"""
We used MOSEK to speed up computations. We recommend using MOSEK, if optimization over large systems
is being done. In this example we obtain series of lower bound indexed by k, where the tightness of
bound increases with k. 
"""
print("k=4:",en.gekppt(rho,[2,2,2],4,solversdp="MOSEK"))
"""
MOSEK often find solutions with higher then desired accuracy.
You can modify MOSEK parameters, inside the code. But if you
do this, you should really know what you are doing.
"""

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
We demostrate different usage of parameters for upperbip.
Proper parametr usage influences performance of the functions.
"""

print(en.upperbip(rho,[4,2],iteramax=1000))
"""
The algorithm depends on initial separable decomposition, which is random.
For that reason, upperbip with the same parameters can give different result
"""
print(en.upperbip(rho,[4,2],iteramax=1000))
"""
We can control accuracy and time of computations, by iteramax, dif.
For qs, sqs see several_similar_states_for_upper_bound
"""

print(en.upperbip(rho,[4,2],dif=10**(-5),iteramax=4500))
"""
Too little dif might cause algorithm to stop before reaching accurate bound.
In each algorithm step we increase fidelity of input rho and separable decomposition. Algorithm stops
when increase in fidelity falls below dif. For that reason setting too little dif might cause too
early stop. 
"""
print(en.upperbip(rho,[4,2],dif=10**(-9),iteramax=4500))
"""
iteramax sets maximal number of iteration steps. If number of iterations exceeds iteramax function
stops computations. Setting too little iteramax might cause inaccurate result. 
"""

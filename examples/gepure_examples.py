import qutip
import numpy as np
import entcalcpy as en

"""
In this script we show some examples of usage of gepure with optional parameters.
We also discuss accuracy of computations.
"""

ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.unit()
print("3qubits",en.ge_pure(ghz,[2,2,2]))
"""
The second position at the output, the error of relaxing separability
condition to PPT is 0. It is due to the fact that we optimized over
two-qubit system and in this case set of separable states is equal
to set of PPT states. For higher dimesnions there is an
estimation error
caused by difference between separable and PPT set of states.
"""

ghz5q=qutip.basis(32,0)+qutip.basis(32,31)
ghz5q=ghz5q.unit()
print("5qubits",en.ge_pure(ghz5q,[2,2,2,2,2]))



x=qutip.rand_ket(32)
print("5qubits",en.ge_pure(x,[2,2,2,2,2]))


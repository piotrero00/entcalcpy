import qutip
import numpy as np
import entcalcpy as en

"""
In this script we show some examples of usage of gepure with optional parameters.
We also discuss accuracy of computations.
"""

ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.unit()
print(en.ge_pure(ghz,[2,2,2]))


import qutip
import numpy as np
import entcalcpy as en

"""
In this script we show some examples of usage of uppermult with optional parameters.
We also discuss accuracy of computations.
"""

ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.unit()
ghz=ghz.proj()



results=[]

for i in range(20):
    res=en.uppersame(ghz,[2,2,2])
    results.append(res)



print(min(results),max(results))
"""
It is known that the geometric entanglement of GHZ state is 0.5. We obtain results around 10**(-9) greater and lower.
Results slightly lower that 0.5 might be surprising, because uppersame should output the upper bound. It happens because
machine precision accuracy. Since we are taking square roots of machine precision 10**(-16), we might encounter 10**(-8) error in
results. One needs to keep it in mind that accuracy is around 10**(-8) and upper bound can be this amount higher than output of the
uppersame. If one have some doubts about the result, one can check correctness of separable decomposition. If one uses optional
argument dec=True, the separable decomposition is also returned. One can then check that it is true separable decomposition
"""

res=en.uppermult(ghz,[2,2,2],dec=True)

print("Upper bound:",res[0])
print("Probability distribution",res[1])
print("Kets in each column:", res[2])

w=qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4)
w=w.unit()
w=w.proj()
print("Examples for GHZ-W mixture:")
rho=0.6*ghz+0.4*w



b=en.uppermult(rho,[2,2,2])
print("Default values result:", b)

res=en.uppermult(rho,[2,2,2],r=4)
print("r=4:",res)
"""
 We accelarate computations by taking smaller r. Keep in mind that we can decrease accuracy by doing this.
 By Carathéodory's theorem we are guaranteed that optimal separable decomposition has r=d**(2), where d is number of dimensions of rho.
 Parameters, sepitera, dif, iteramax determines accuracy and speed of computations. 
"""

res=en.uppermult(rho,[2,2,2],r=4,sepitera=25)
print("sepitera=25,r=4:",res)
"""
 Increasing sepitera we can reduce number of iterations before convergance.
 But be careful, too great sepitera slow downs computations without increasing accuracy.
"""


res=en.uppermult(rho,[2,2,2],r=4,dif=10**(-5))
print("dif=10**(-5),r=4:",res)

"""
Smaller dif makes computations faster, but result might not be accurate. The function terminates if fidelity update between subsequent iterations is smaller than dif.
If function converges slowly, putting too little dif might result in unprecise bound.
"""

res=en.uppermult(rho,[2,2,2],iteramax=500)
print("dims=[2,2,2],iteramax=500",res)#Note that with dec=False the output is a float number, with dec=True it is list with 3 elements

"""
iteramax determines maximal number of iterations. It prevents too long computations. Together with dif it can bu used
to manipulate precision of computations.
"""
res=en.uppermult(rho,[4,2],r=10,iteramax=500)
print("dims=[4,2],iteramax=500:",res)
"""
uppermult is also able to compute upperbound for systems with different dimensions
For a bipartite system upperbip does it more efficiently. For multipartite,
uppermult is more efficient
"""

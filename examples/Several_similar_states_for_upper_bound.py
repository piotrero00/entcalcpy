import qutip
import numpy as np
import entcalcpy as en

"""
We demonstrate how to use uppersame to accelerate computations of
entanglement upper bound for similar states. Assume we want to compute
the upper bound for some spin chains with changing inverse temperature

"""

#Define set-up
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
bet=0.6
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
HH=-H*bet
r=HH.expm()

#print(r)
state=r/r.tr()
#print(lo)

results=[]
res=en.uppersame(state,[2,2,2],iteramax=5,dif=10**(-8),dec=True)

for i in range(60,100,5):
    
    bet=i/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)

    HH=-H*bet
    r=HH.expm()
    
    state=r/r.tr()
        
    res=en.uppersame(state,[2,2,2],iteramax=4000,dif=10**(-8)) #We compute upper bound for each state, everytime starting from random decomposition

    results.append(res)


"""
We can accelerate computations by starting from state that was output of the previous state. Doing that we do not have random decomposition
at the beggining and we converge faster. We can put lower iteration number.
"""
bet=0.6
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)

HH=-H*bet
r=HH.expm()
#print(r)
state=r/r.tr()
#print(lo)
result2=[]
res=en.uppersame(state,[2,2,2],iteramax=4000,dif=10**(-9),dec=True) #First computation must be done from random decomposition


result2.append(res[0])   #First output is the upper bound


il=0
for i in range(65,100,5):
    il+=1
    bet=i/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)

    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()       
    """
        Now we will start with decomposition, which optimized previous computations. Since change in state is small,
        we expect the output to converge faster.
    """
    res=en.uppersame(state,[2,2,2],iteramax=500,dif=10**(-8),dec=True,qs=res[1],sqs=res[2])  #Secend output is probability distribution, third output is state ensamble

    
    result2.append(res[0])


"""
    The output res[2] consist of matrix which columns are subsequent kets of the separable decomposition.
    res[1] outputs probablilities corresponding to these kets
"""
    
    




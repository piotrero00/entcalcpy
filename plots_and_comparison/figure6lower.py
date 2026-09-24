#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np


#Lower bound for figure 6



sy1=qutip.tensor(qutip.sigmay(),qutip.qeye(2),qutip.qeye(2))
sy2=qutip.tensor(qutip.qeye(2),qutip.sigmay(),qutip.qeye(2))
sy3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmay())

sx1=qutip.tensor(qutip.sigmax(),qutip.qeye(2),qutip.qeye(2))
sx2=qutip.tensor(qutip.qeye(2),qutip.sigmax(),qutip.qeye(2))
sx3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmax())

sz1=qutip.tensor(qutip.sigmaz(),qutip.qeye(2),qutip.qeye(2))
sz2=qutip.tensor(qutip.qeye(2),qutip.sigmaz(),qutip.qeye(2))
sz3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmaz())
#beta=0.6
resultsp=[]
bet=0.6
for hi in range(-300,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure6lower0--60.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')


#beta=1
resultsp=[]
bet=1
for hi in range(-300,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure6lower1.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')

#beta=10
resultsp=[]
bet=10
for hi in range(-300,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure6lower10.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')



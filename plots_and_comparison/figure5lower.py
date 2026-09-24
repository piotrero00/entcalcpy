#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np


#Lower bound for figure 5


#Computation of the first part
sy1=qutip.tensor(qutip.sigmay(),qutip.qeye(2),qutip.qeye(2))
sy2=qutip.tensor(qutip.qeye(2),qutip.sigmay(),qutip.qeye(2))
sy3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmay())

sx1=qutip.tensor(qutip.sigmax(),qutip.qeye(2),qutip.qeye(2))
sx2=qutip.tensor(qutip.qeye(2),qutip.sigmax(),qutip.qeye(2))
sx3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmax())

sz1=qutip.tensor(qutip.sigmaz(),qutip.qeye(2),qutip.qeye(2))
sz2=qutip.tensor(qutip.qeye(2),qutip.sigmaz(),qutip.qeye(2))
sz3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmaz())
resultsp=[]

for b in range(30,65,5):
    bet=b/100
    J=-1
    h=1
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure5upper1aa.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')


resultsp=[]

for b in range(30,65,5):
    bet=b/100
    J=-1
    h=2
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure5upper2aa.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')


resultsp=[]

for b in range(30,65,5):
    bet=b/100
    J=-1
    h=1.5
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


with open ('figure5upper1-5aa.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')




np.random.seed(12345678)


#h=1
resultsp=[]
for b in range(60,600,5):
    bet=b/100
    J=-1
    h=1
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])


for b in range(600,1010,10):
    bet=b/100
    J=-1
    h=1
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])

with open ('figure5upper1.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')



#h=1.5
resultsm=[]
for b in range(60,600,5):
    bet=b/100
    J=-1
    h=1.5
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsm.append(lower[0])
    print(lower[0])


for b in range(600,1010,10):
    bet=b/100
    J=-1
    h=1.5
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsm.append(lower[0])
    print(lower[0])


with open ('figure5lower1-5.txt', 'w') as filehandle:
    for lis in resultsm:
        filehandle.write(f'{lis}\n')
        




#h=2
resultso=[]
for b in range(60,600,5):
    bet=b/100
    J=-1
    h=2
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultso.append(lower[0])
    print(lower[0])


for b in range(600,1010,10):
    bet=b/100
    J=-1
    h=2
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultso.append(lower[0])
    print(lower[0])


with open ('figure5lower2.txt', 'w') as filehandle:
    for lis in resultso:
        filehandle.write(f'{lis}\n')


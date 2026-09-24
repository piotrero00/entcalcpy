#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en


#Script that generates lower bounds for figure 4



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

for b in range(640,760,5):
    bet=b/1000
    J=1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsp.append(lower[0])
    print(lower[0])
    

with open ('figure4lowerj1.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')

resultsm=[]

for b in range(640,760,5):
    bet=b/1000
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsm.append(lower[0])
    print(lower[0])
    

with open ('figure4lowerj-1.txt', 'w') as filehandle:
    for lis in resultsm:
        filehandle.write(f'{lis}\n')


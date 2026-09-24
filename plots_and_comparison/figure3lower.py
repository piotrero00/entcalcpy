#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en






sy1=qutip.tensor(qutip.sigmay(),qutip.qeye(2),qutip.qeye(2))
sy2=qutip.tensor(qutip.qeye(2),qutip.sigmay(),qutip.qeye(2))
sy3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmay())

sx1=qutip.tensor(qutip.sigmax(),qutip.qeye(2),qutip.qeye(2))
sx2=qutip.tensor(qutip.qeye(2),qutip.sigmax(),qutip.qeye(2))
sx3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmax())

sz1=qutip.tensor(qutip.sigmaz(),qutip.qeye(2),qutip.qeye(2))
sz2=qutip.tensor(qutip.qeye(2),qutip.sigmaz(),qutip.qeye(2))
sz3=qutip.tensor(qutip.qeye(2),qutip.qeye(2),qutip.sigmaz())
#J=1
results=[]

for b in range(60,600,5):
    bet=b/100
    J=1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    results.append(lower[0])
    print(lower[0])
    

for b in range(600,1010,10):
    bet=b/100
    J=1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    results.append(lower[0])
    print(lower[0])


with open ('figure3lowerj1p.txt', 'w') as filehandle:
    for lis in results:
        filehandle.write(f'{lis}\n')



#J=-1
resultsm=[]
for b in range(60,600,5):
    bet=b/100
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
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

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    lower=en.ge_mixed_gr(state,[2,2,2],solversdp="MOSEK")

    resultsm.append(lower[0])
    print(lower[0])


with open ('figure3lowerj-1a.txt', 'w') as filehandle:
    for lis in resultsm:
        filehandle.write(f'{lis}\n')




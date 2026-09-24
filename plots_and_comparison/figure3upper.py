#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np



np.random.seed(123456)

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
b=60
bet=b/100
J=1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)

results.append(upper3[0])


for b in range(65,600,5):
    bet=b/100
    J=1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        qs=upper3[1],
        sqs=upper3[2],
        sepitera=15,
        dec=True)

    results.append(upper3[0])
    print(upper3[0])
    

for b in range(600,1010,10):
    bet=b/100
    J=1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        qs=upper3[1],
        sqs=upper3[2],
        sepitera=15,
        dec=True)

    results.append(upper3[0])
    print(upper3[0])


with open ('figure3upperpp.txt', 'w') as filehandle:
    for lis in results:
        filehandle.write(f'{lis}\n')


np.random.seed(694217)
#J=-1


results=[]
b=60
bet=b/100
J=-1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
results.append(upper3[0])

resultsm=[]
for b in range(60,600,5):
    bet=b/100
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        qs=upper3[1],
        sqs=upper3[2],
        sepitera=15,
        dec=True)

    resultsm.append(upper3[0])
    print(upper3[0])


for b in range(600,1010,10):
    bet=b/100
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
    HH=-H*bet
    r=HH.expm()
    #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        qs=upper3[1],
        sqs=upper3[2],
        sepitera=15,
        dec=True)

    resultsm.append(upper3[0])
    print(upper3[0])


with open ('figure3uppermm.txt', 'w') as filehandle:
    for lis in resultsm:
        filehandle.write(f'{lis}\n')



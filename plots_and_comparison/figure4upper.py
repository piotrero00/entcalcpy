#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np

#Script that generates lower bounds for figure 4


np.random.seed(12345678)

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
b=640
bet=b/1000
J=1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for b in range(645,760,5):
    bet=b/1000
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
        sepitera=12,
        dec=True)

    resultsp.append(upper3[0])
    print(upper3[0])
    

with open ('figure4upperpp.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')

np.random.seed(81276345)
resultsm=[]
b=640
bet=b/1000
J=-1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1)
    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)

resultsm.append(upper3[0])

for b in range(640,760,5):
    bet=b/1000
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
        sepitera=12,
        dec=True)

    resultsm.append(upper3[0])
    print(upper3[0])
    

with open ('figure4uppermm.txt', 'w') as filehandle:
    for lis in resultsm:
        filehandle.write(f'{lis}\n')


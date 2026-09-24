import qutip
import entcalcpy as en
import numpy as np


#upper bound for figure 6


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

#beta=0.6
resultsp=[]
bet=0.6

hi=-300
h=hi/100
J=-1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for hi in range(-295,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
    HH=-H*bet
    r=HH.expm()
        #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        dec=True,sepitera=12)
    resultsp.append(upper3[0])
    print(upper3[0])

with open ('figure6upper0--60a.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')


#beta=1
np.random.seed(87654321)
resultsp=[]
bet=1

hi=-300
h=hi/100
J=-1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for hi in range(-295,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
    HH=-H*bet
    r=HH.expm()
        #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        dec=True,sepitera=12)
    resultsp.append(upper3[0])
    print(upper3[0])

with open ('figure6upper1a.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')

#beta=10
np.random.seed(65432178)
resultsp=[]
bet=10

hi=-300
h=hi/100
J=-1

H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for hi in range(-295,305,5):
    
    J=-1
    h=hi/100
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
    HH=-H*bet
    r=HH.expm()
        #print(r)
    state=r/r.tr()
    upper3=en.uppersame(state,[2,2,2],iteramax=500,
        dif=1e-9,
        dec=True,sepitera=12)
    resultsp.append(upper3[0])
    print(upper3[0])

with open ('figure6upper2a.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')



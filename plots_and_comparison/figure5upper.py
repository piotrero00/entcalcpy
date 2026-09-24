#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np


#Lower bound for figure 5


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
b=30
bet=b/100
J=-1
h=1
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)


    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])



#warm start



np.random.seed(12345678)



#h=1

resultsp=[]
b=30
bet=b/100
J=-1
h=1
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for b in range(35,600,5):
    bet=b/100
    h=1
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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
    

for b in range(600,1010,10):
    bet=b/100
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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


with open ('figure5upper1b.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')



#h=1.5
np.random.seed(69123457)
resultsp=[]
b=30
bet=b/100
J=-1
h=1.5
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for b in range(35,600,5):
    bet=b/100
    h=1.5
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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
    

for b in range(600,1010,10):
    bet=b/100
    J=-1
    h=1.5
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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


with open ('figure5upper1-5b.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')




np.random.seed(87654321)
#h=2
resultsp=[]
b=30
bet=b/100
J=-1
h=2
H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)

    
HH=-H*bet
r=HH.expm()
    #print(r)
state=r/r.tr()
upper3=en.uppersame(state,[2,2,2],iteramax=5000,
    dif=1e-9,
    dec=True)
resultsp.append(upper3[0])

for b in range(35,600,5):
    bet=b/100
    h=2
    J=-1

    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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
    

for b in range(600,1010,10):
    bet=b/100
    J=-1
    h=2
    H=-J/2*(sy1*sy2+sy2*sy3+sy3*sy1+sx1*sx2+sx2*sx3+sx3*sx1+sz1*sz2+sz2*sz3+sz3*sz1)+h*(sz1+sz2+sz3)
    
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


with open ('figure5upper2b.txt', 'w') as filehandle:
    for lis in resultsp:
        filehandle.write(f'{lis}\n')


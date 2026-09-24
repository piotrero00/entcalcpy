#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en
import numpy as np

#Script for generating data for lower bounds on figure 2

SEED = 123456
np.random.seed(SEED)

#3 qubits
res=[]
ghz=qutip.basis(8,0)+qutip.basis(8,7)
ghz=ghz.proj()
ghz=ghz.unit()

w=qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4)
w=w.proj()
w=w.unit()

rho=ghz
upper3=en.uppersame(
    rho,
    [2,2,2],
    iteramax=5000,
    dif=1e-9,
    dec=True
)
res.append(upper3[0])
for i in range(1,101):
    p=i/100
    rho=p*w+(1-p)*ghz
    upper3=en.uppersame(
        rho,
        [2,2,2],
        iteramax=500,
        dif=1e-9,
        qs=upper3[1],
        sqs=upper3[2],
        sepitera=12,
        dec=True
    )
    res.append(upper3[0])
    print(upper3[0])

with open ('upper3ghzww.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')






#4qubits

np.random.seed(234567)

res=[]
ghz=qutip.basis(16,0)+qutip.basis(16,15)
ghz=ghz.proj()
ghz=ghz.unit()

w=qutip.basis(16,1)+qutip.basis(16,2)+qutip.basis(16,4)+qutip.basis(16,8)
w=w.proj()
w=w.unit()

rho=ghz
upper4=en.uppersame(
    rho,
    [2,2,2,2],
    iteramax=5000,
    dif=1e-9,
    dec=True,
    r=16,
    sepitera=15
    
)
res.append(upper4[0])
for i in range(1,101):
    p=i/100
    rho=p*w+(1-p)*ghz
    upper4=en.uppersame(
        rho,
        [2,2,2,2],
        iteramax=500,
        dif=1e-9,
        qs=upper4[1],
        sqs=upper4[2],
        sepitera=12,
        dec=True,
        r=16
    )
    res.append(upper4[0])
    print(upper4[0])

with open ('upper4ghzww.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')




#5qubits

np.random.seed(345678)
res=[]
ghz=qutip.basis(32,0)+qutip.basis(32,31)
ghz=ghz.proj()
ghz=ghz.unit()

w=qutip.basis(32,1)+qutip.basis(32,2)+qutip.basis(32,4)+qutip.basis(32,8)+qutip.basis(32,16)
w=w.proj()
w=w.unit()
rho=ghz
upper5=en.uppersame(
    rho,
    [2,2,2,2,2],
    iteramax=5000,
    dif=1e-9,
    dec=True,r=16
)
res.append(upper5[0])
for i in range(1,101):
    p=i/100
    rho=p*w+(1-p)*ghz
    upper5=en.uppersame(
        rho,
        [2,2,2,2,2],
        iteramax=500,
        dif=1e-9,
        qs=upper5[1],
        sqs=upper5[2],
        sepitera=12,
        dec=True,
        r=16
    )
    res.append(upper5[0])
    print(upper5[0])

with open ('upper5ghzww.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')



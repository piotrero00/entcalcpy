#This script was created in entcalcpy version 0.1.3


import qutip
import entcalcpy as en


#Script for generating data for lower bounds on figure 2



#3 qubits
res=[]
ghz=qutip.basis(8,0)+qutip.basis(8,7)

ghz=ghz.unit()

w=qutip.basis(8,1)+qutip.basis(8,2)+qutip.basis(8,4)

w=w.unit()

for i in range(101):
    p=i/100
    lower3=en.ge_mixed_ra_gr([ghz,w],[p,1-p],[2,2,2],solversdp="MOSEK")
    res.append(lower3[0])
    print(lower3[0])

with open ('lower3ghzw.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')







#4qubits
res=[]
ghz=qutip.basis(16,0)+qutip.basis(16,15)

ghz=ghz.unit()

w=qutip.basis(16,1)+qutip.basis(16,2)+qutip.basis(16,4)+qutip.basis(16,8)

w=w.unit()
for i in range(101):
    p=i/100
    lower4=en.ge_mixed_ra_gr([ghz,w],[p,1-p],[2,2,2,2],solversdp="MOSEK")
    res.append(lower4[0])
    print(lower4[0])

with open ('lower4ghzw.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')

#5qubits
res=[]
ghz=qutip.basis(32,0)+qutip.basis(32,31)

ghz=ghz.unit()

w=qutip.basis(32,1)+qutip.basis(32,2)+qutip.basis(32,4)+qutip.basis(32,8)+qutip.basis(32,16)

w=w.unit()
for i in range(101):
    p=i/100
    lower5=en.ge_mixed_ra_gr([ghz,w],[p,1-p],[2,2,2,2,2],solversdp="MOSEK")
    res.append(lower5[0])
    print(lower5[0])

with open ('lower5ghzw.txt', 'w') as filehandle:
    for lis in res:
        filehandle.write(f'{lis}\n')

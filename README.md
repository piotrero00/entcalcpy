# entcalcpy
entcalcpy is a Python package for computing the geometric entanglement of a given quantum state. 
It does it by computing lower and upper bounds for the geometric entanglement.
These bounds are often close to each other allowing us to estimate the value of the geometric entanglement in a given quantum state.
## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Getting started](#getting-started)
- [Issues](#issues)
- [Acknowledgment](#acknowledgment)
- [License](#license)
- [Citing](#citing)
# Installation
```bash
pip install entcalcpy
```
# Dependencies
This package require the following Python packages:
- `numpy` ≥ 2.24  
- `scipy` ≥ 1.15.2  
- `cvxpy` ≥ 1.6.4  
- `qutip` ≥ 5.1.0
# Getting started
To get started with entcalcpy we recommend checking the examples section. The documentation is written in docstrings.
For example, the following code computes the lower bound of the geometric entanglement of a random quantum state.
```python
import qutip
import entcalcpy as en
rho=qutip.rand_dm(8)
print(en.uppersame(rho,[2,2,2])) #second argument specifies that we have a 3-qubit state
print(en.upperbip(rho,[4,2])) #here we have a 4x2 bipartite state
```
Similarly we can compute the lower bound
```python
import qutip
import entcalcpy as en
rho=qutip.rand_dm(8)
print(en.ge_mixed_gr(rho,[2,2,2])) #second argument specifies that we have a 3-qubit state
print(en.ge_mixed_gr(rho,[4,2])) #here we have a 4x2 bipartite state
```
Lower bound can be computed using 4 different methods. For more information see examples.
# Issues
If you find any issue, we encourage you to report it using github or via e-mail on adress maspiotr00@gmail.com
# Acknowledgment
Special thanks to Krystyna Mordzińska for coming up with the package name after a creative brainstorming session.
# License
Fill this in
# Citing
If you use entcalcpy for academic work, please cite us. 

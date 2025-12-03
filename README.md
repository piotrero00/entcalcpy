# entcalcpy
entcalcpy is a Python package for computing the geometric entanglement of a given quantum state. 
It works by computing lower and upper bounds for the geometric entanglement.
These bounds are often close to each other, allowing us to estimate the value of the geometric entanglement in a given quantum state.
## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Getting started](#getting-started)
- [Issues](#issues)
- [Acknowledgment](#acknowledgment)
- [License](#license)
- [Citing](#citing)
## Installation
```bash
pip install entcalcpy
```
## Dependencies
This package requires the following Python packages:
- `numpy` ≥ 2.2.4  
- `scipy` ≥ 1.15.2  
- `cvxpy` ≥ 1.6.4  
- `qutip` ≥ 5.1.0
## Getting started
To get started with entcalcpy, we recommend checking the examples section. The documentation is written in docstrings.
For example, the following code computes the upper bound of the geometric entanglement of a random quantum state.
```python
import qutip
import entcalcpy as en
rho=qutip.rand_dm(8)
print(en.uppersame(rho,[2,2,2])) #second argument specifies that we have a 3-qubit state
print(en.upperbip(rho,[4,2])) #here we have a 4x2 bipartite state
```
Similarly, we can compute the lower bound
```python
import qutip
import entcalcpy as en
rho=qutip.rand_dm(8)
print(en.ge_mixed_gr(rho,[2,2,2])) #second argument specifies that we have a 3-qubit state
print(en.ge_mixed_gr(rho,[4,2])) #here we have a 4x2 bipartite state
```
The lower bound can be computed using four different methods.
## ge_mixed_gr vs ge_mixed_ra_gr
ge_mixed_gr and ge_mixed_sm as a first step computes a purification of the input state. They do it by so-called canonical purification. 
ge_mixed_ra_gr takes input state in the form of orthogonal decomposition. Thanks to it, it can compute lower-dimensional purification and compute
 lower boudnd more efficiently.
 For example, if ge_mixed_gr takes n-qubit state as input, it must optimize over matrix with dimesnions $2^{2n}\times 2^{2n}$ matrices. If the state is rank-2, ge_mixed_ra_gr optimizes
 over matrix with dimesnions $2^{n+1}\times 2^{n+1}$. Thanks to this, it can compute entanglement for more qubits and can make it much faster.
## Solvers
entcalcpy can use two solvers: "SCS" and "MOSEK". "SCS" is installed when cvxpy is installed. For some problems, it may need a long time to find a solution. If the solution is not found within given iteration limit, it return message that the solution might be inaccurate. In that scenario, the result might be higher
than a lower bound. If this happens, we recommend using "MOSEK". It is not a free solver, but one can use it for free for academic purposes.
## Issues
If you find any issues, we encourage you to report them via GitHub or by emailing maspiotr00@gmail.com.
## Acknowledgment
Special thanks to Krystyna Mordzińska for coming up with the package name after a creative brainstorming session.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Citing
If you use entcalcpy in academic work, please cite this package.

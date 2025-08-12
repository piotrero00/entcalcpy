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
For example, the following code computes the lower bound of the geometric entanglement of a random quantum state.
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
## Issues
If you find any issues, we encourage you to report them via GitHub or by emailing maspiotr00@gmail.com.
## Acknowledgment
Special thanks to Krystyna Mordzińska for coming up with the package name after a creative brainstorming session.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
## Citing
If you use entcalcpy in academic work, please cite this package.

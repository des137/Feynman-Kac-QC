# Feynman-Kac Quantum Computing

[![arXiv](https://img.shields.io/badge/arXiv-2106.10066-b31b1b.svg)](https://arxiv.org/abs/2106.10066)
[![Quantum Journal](https://img.shields.io/badge/Quantum-10.22331%2Fq--2022--06--07--730-blue)](https://quantum-journal.org/papers/q-2022-06-07-730/)

Source code for the article: [A variational quantum algorithm for the Feynman-Kac formula](https://quantum-journal.org/papers/q-2022-06-07-730/).

## Overview

This repository implements a variational quantum algorithm for solving the Feynman-Kac formula, which is a fundamental tool in stochastic processes and mathematical finance. The algorithm uses quantum computing techniques to efficiently compute expectations of functionals of stochastic processes.

The implementation includes:
- Quantum gate operations (Hadamard, rotation gates, controlled gates)
- Natural gradient descent optimization for variational quantum algorithms
- Parameter shift rule for gradient computation
- Utility functions for quantum state manipulation
- Example implementations and feasibility tests

## Project Structure

```
Feynman-Kac-QC/
├── src/                          # Source code directory
│   ├── __init__.py              # Package initialization
│   ├── gates.py                 # Quantum gate implementations (CX, H, rx, ry, rz, cry, p)
│   ├── utils.py                 # Utility functions (tensor products, expectation values)
│   ├── natgrad.py               # Natural gradient descent implementation
│   ├── parameter_shift.py       # Parameter shift rule for gradients
│   ├── topography.py            # Energy landscape visualization
│   └── main.py                  # Main execution script with example usage
├── feasibility_test.ipynb       # Jupyter notebook testing the approach feasibility
├── README.md                    # This file
└── .gitignore                   # Git ignore rules

```

## Installation

### Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook (optional, for running the feasibility test notebook)
- Qiskit (optional, used in the feasibility test notebook)
- scikit-learn (optional, used in the feasibility test notebook)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/des137/Feynman-Kac-QC.git
cd Feynman-Kac-QC
```

2. Install dependencies:
```bash
pip install numpy matplotlib jupyter qiskit scikit-learn
```

## Usage

### Running the Main Example

The main script demonstrates the variational quantum algorithm on a simple Hamiltonian:

```bash
python -m src.main
```

This will:
1. Define a simple quantum ansatz with parametrized gates
2. Set up a sample Hamiltonian
3. Evolve the system using natural gradient descent
4. Output the evolution trajectory

### Using Individual Modules

You can import and use individual components:

```python
from src.gates import rx, ry, rz, cry, CX, H
from src.utils import tensorprod, expectation
from src.natgrad import nat_grad
import numpy as np

# Define quantum states
Zero = np.array([1, 0]).reshape(-1, 1)
One = np.array([0, 1]).reshape(-1, 1)

# Create a quantum state using tensor products
state = tensorprod(Zero, One)

# Apply quantum gates
rotated_state = rx(np.pi/4) @ Zero

# Compute expectation values
H_operator = np.array([[1, 0], [0, -1]])
exp_val = expectation(H_operator, rotated_state)
```

### Exploring the Feasibility Test

The `feasibility_test.ipynb` notebook contains a detailed exploration of the algorithm's feasibility, including:
- Implementation of a ZZ-feature map ansatz
- Comparison with Qiskit's implementation
- Performance benchmarking
- Visualization of quantum states

To run the notebook:
```bash
jupyter notebook feasibility_test.ipynb
```

## Key Components

### Quantum Gates (`src/gates.py`)

Implements standard quantum gates:
- **CX**: Controlled-X (CNOT) gate
- **H**: Hadamard gate
- **rx, ry, rz**: Single-qubit rotation gates
- **cry**: Controlled-RY gate
- **p**: Phase gate

### Utility Functions (`src/utils.py`)

- `tensorprod(*args)`: Compute tensor product of multiple matrices
- `conj_tp(qobj)`: Conjugate transpose
- `bra_ket(state1, state2)`: Inner product of quantum states
- `expectation(H, state)`: Compute expectation value of Hamiltonian

### Natural Gradient (`src/natgrad.py`)

Implements the quantum natural gradient method for optimization, which uses the quantum geometric tensor (Fubini-Study metric) to improve convergence.

### Parameter Shift (`src/parameter_shift.py`)

Implements the parameter shift rule for computing gradients of quantum circuits, enabling hardware-compatible gradient computation.

## Algorithm

The variational quantum algorithm for Feynman-Kac formula works as follows:

1. **Initialize** a parameterized quantum state (ansatz)
2. **Compute** the natural gradient using the quantum Fisher information matrix
3. **Update** parameters using gradient descent
4. **Iterate** until convergence

The natural gradient provides a more efficient optimization landscape compared to standard gradient descent by accounting for the geometry of the quantum state space.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kyriienko2022variational,
  title={A variational quantum algorithm for the Feynman-Kac formula},
  author={Kyriienko, Oleksandr and Paine, Annie E and Elfving, Vincent E},
  journal={Quantum},
  volume={6},
  pages={730},
  year={2022},
  publisher={Verlag der Österreichischen Akademie der Wissenschaften}
}
```

## License

Please refer to the original paper for licensing information.

## Contributing

This repository contains the source code for a published research paper. For questions or discussions about the implementation, please open an issue.

## Contact

For questions about the algorithm or implementation, please refer to the original paper or open an issue in this repository.

"""Linear-algebra utilities for quantum state manipulation.

Provides tensor products, conjugate transpose (dagger), inner products,
and expectation values.
"""

from functools import reduce

import numpy as np


def tensorprod(*args):
    """Kronecker (tensor) product of an arbitrary number of matrices."""
    return reduce(np.kron, args)


def dagger(qobj):
    """Conjugate transpose (Hermitian adjoint) of a matrix or vector."""
    return qobj.conj().T


def bra_ket(state1, state2):
    """Inner product <state1|state2>."""
    return (dagger(state1) @ state2)[0, 0]


def expectation(hamiltonian, state):
    """Real expectation value <state|H|state>."""
    return np.real(bra_ket(state, hamiltonian @ state))

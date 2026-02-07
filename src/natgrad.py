"""Natural gradient descent via finite differences.

Computes the quantum natural gradient using the Fubini-Study metric
estimated with finite-difference derivatives of the ansatz.
"""

import numpy as np

from src.utils import bra_ket, dagger


def nat_grad(theta, hamiltonian, ansatz, h=0.001):
    """Return the natural gradient vector for parameter vector *theta*.

    Parameters
    ----------
    theta : array_like
        Current variational parameters.
    hamiltonian : ndarray
        Hamiltonian matrix.
    ansatz : callable
        Maps parameter vector to a quantum state (column vector).
    h : float, optional
        Finite-difference step size (default 0.001).
    """
    theta = np.asarray(theta, dtype=float)
    n_params = len(theta)

    # Finite-difference derivative vectors
    psi = ansatz(theta)
    grad_vecs = np.empty((n_params, *psi.shape), dtype=complex)
    for i in range(n_params):
        shift = np.zeros(n_params)
        shift[i] = h
        grad_vecs[i] = (ansatz(theta + shift) - psi) / h

    # Fubini-Study metric tensor
    projector = psi @ dagger(psi)
    M = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(n_params):
            M[i, j] = np.real(
                bra_ket(grad_vecs[i], grad_vecs[j])
                - bra_ket(grad_vecs[i], projector @ grad_vecs[j])
            )

    # Energy gradient
    H_psi = hamiltonian @ psi
    C = np.array([bra_ket(grad_vecs[i], H_psi) for i in range(n_params)])

    return np.real(np.linalg.pinv(M) @ C)

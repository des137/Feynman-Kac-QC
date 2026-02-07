"""Natural gradient descent via the parameter-shift rule.

Computes the quantum natural gradient using analytic pi/2-shift
derivatives of the ansatz instead of finite differences.
"""

import numpy as np

from src.utils import bra_ket, dagger


def parameter_shift(theta, hamiltonian, ansatz):
    """Return the natural gradient vector using the parameter-shift rule.

    Parameters
    ----------
    theta : array_like
        Current variational parameters.
    hamiltonian : ndarray
        Hamiltonian matrix.
    ansatz : callable
        Maps parameter vector to a quantum state (column vector).
    """
    theta = np.asarray(theta, dtype=float)
    n_params = len(theta)

    # Parameter-shift derivative vectors
    psi = ansatz(theta)
    grad_vecs = np.empty((n_params, *psi.shape), dtype=complex)
    for i in range(n_params):
        shift = np.zeros(n_params)
        shift[i] = np.pi / 2
        grad_vecs[i] = (ansatz(theta + shift) - ansatz(theta - shift)) / 2

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

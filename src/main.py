"""Feynman-Kac imaginary-time evolution driver.

Provides :func:`evolve` which propagates variational parameters forward
in imaginary time using the natural gradient, and a runnable example
under ``__main__``.
"""

import numpy as np

from src.gates import cry, rx
from src.natgrad import nat_grad
from src.utils import tensorprod


def evolve(initial_thetas, dt, timesteps, hamiltonian, ansatz):
    """Imaginary-time evolution of variational parameters.

    Parameters
    ----------
    initial_thetas : array_like
        Starting parameter values.
    dt : float
        Imaginary-time step size.
    timesteps : int
        Number of evolution steps.
    hamiltonian : ndarray
        Hamiltonian matrix.
    ansatz : callable
        Maps parameter vector to a quantum state (column vector).

    Returns
    -------
    ndarray of shape (timesteps, n_params)
        Parameter trajectory over all time steps.
    """
    n_params = len(initial_thetas)
    trajectory = np.zeros((timesteps, n_params))
    trajectory[0] = initial_thetas
    for step in range(1, timesteps):
        grad = nat_grad(trajectory[step - 1], hamiltonian, ansatz).T
        trajectory[step] = trajectory[step - 1] - dt * grad
    return trajectory


if __name__ == "__main__":
    Zero = np.array([1, 0]).reshape(-1, 1)
    I2 = np.eye(2)

    def ansatz(theta):
        theta = np.asarray(theta)
        init = tensorprod(Zero, Zero)
        layer1 = tensorprod(rx(theta[0]), I2)
        layer2 = cry(theta[1])
        return layer2 @ layer1 @ init

    hamiltonian = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
    ])

    t_init = np.array([np.pi / 2, 0.05])
    dt = 0.1
    timesteps = 1000

    evolve(t_init, dt, timesteps, hamiltonian, ansatz)

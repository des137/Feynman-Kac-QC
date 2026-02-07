"""Energy landscape visualization.

Generates a 2-D heatmap of expectation values over a grid of
two variational parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.utils import expectation


def plot_energy_landscape(hamiltonian, ansatz, theta_range=(-np.pi, np.pi),
                          resolution=0.1):
    """Plot the energy landscape for a two-parameter ansatz.

    Parameters
    ----------
    hamiltonian : ndarray
        Hamiltonian matrix.
    ansatz : callable
        Maps a 2-element parameter vector to a quantum state (column vector).
    theta_range : tuple of float, optional
        (min, max) range for both parameter axes (default (-pi, pi)).
    resolution : float, optional
        Grid spacing in radians (default 0.1).
    """
    theta1 = np.arange(theta_range[0], theta_range[1], resolution)
    theta2 = np.arange(theta_range[0], theta_range[1], resolution)

    energy = np.empty((len(theta1), len(theta2)))
    for i in range(len(theta1)):
        for j in range(len(theta2)):
            energy[i, j] = expectation(hamiltonian, ansatz([theta1[i], theta2[j]]))

    plt.imshow(energy.T, origin="lower",
               extent=[theta_range[0], theta_range[1],
                       theta_range[0], theta_range[1]])
    plt.colorbar(label="Energy")
    plt.xlabel("theta_1")
    plt.ylabel("theta_2")
    plt.title("Energy Landscape")
    plt.show()

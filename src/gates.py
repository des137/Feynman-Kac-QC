"""Quantum gate definitions.

Provides standard single- and two-qubit gates used in variational circuits:
CX (CNOT), Hadamard, rotation gates (Rx, Ry, Rz), controlled-Ry, and phase.
"""

import numpy as np

CX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
])

H = (1 / np.sqrt(2)) * np.array([
    [1,  1],
    [1, -1],
])


def rx(theta):
    """Rotation about the X axis by angle *theta*."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [    c, -1j * s],
        [-1j * s,     c],
    ])


def ry(theta):
    """Rotation about the Y axis by angle *theta*."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s,  c],
    ])


def rz(theta):
    """Rotation about the Z axis by angle *theta*."""
    return np.array([
        [np.exp(-1j * theta / 2),                      0],
        [                      0, np.exp(1j * theta / 2)],
    ])


def cry(theta):
    """Controlled-Ry gate."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [1, 0,  0,  0],
        [0, 1,  0,  0],
        [0, 0,  c, -s],
        [0, 0,  s,  c],
    ])


def p(phi):
    """Phase gate with angle *phi*."""
    return np.array([
        [1,              0],
        [0, np.exp(1j * phi)],
    ])

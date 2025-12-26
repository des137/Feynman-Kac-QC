import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from src.utils import expectation

# Need to include HAM and Ansatz

# Parameters
theta1 = np.arange(-pi, pi, 0.1)
theta2 = np.arange(-pi, pi, 0.1)

# Energy topography
Z_tom = np.empty((len(theta1), len(theta2)))
for i in range(len(theta1)):
    for j in range(len(theta2)):
        Z_tom[i, j] = expectation(HAM, Ansatz([theta1[i], theta2[j]]))
plt.imshow(Z_tom.T)
plt.show()

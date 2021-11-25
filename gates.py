CX = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 0, 1], 
    [0, 0, 1, 0]
])

def rx(theta):
    return np.array([
        [    cos(theta/2), -1j*sin(theta/2)],
        [-1j*sin(theta/2),     cos(theta/2)]
    ])

def ry(theta):
    return np.array([
        [cos(theta/2), -sin(theta/2)],
        [sin(theta/2),  cos(theta/2)]
    ])

def rz(theta):
    return np.array([
        [exp(-1j*theta/2),               0],
        [               0, exp(1j*theta/2)]
    ])

def cry(theta):
    cry_ = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0], 
        [0, 0, cos(theta/2), -sin(theta/2)], 
        [0, 0, sin(theta/2), cos(theta/2)]]
    )
    return cry_
    

def tensorprod(*args):
    matrix = 1
    for i in args:
        matrix = np.kron(matrix, i)
    return matrix

def conj_tp(qobj):
    return qobj.conj().T    

def bra_ket(state1, state2):
    return (conj_tp(state1) @ state2)[0]

def expectation(H, state):
    return np.real(bra_ket(state, H @ state))

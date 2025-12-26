import numpy as np
from src.utils import bra_ket, conj_tp

def parameter_shift(theta, H, ansatz_):
    theta = np.array(theta)
    len_th = len(theta)
    
    g = []
    for i in range(len(theta)):
        diff_ = np.zeros(len(theta))
        diff_[i] = np.pi/2
        vec_ = (ansatz_(theta + diff_) - ansatz_(theta - diff_))/2    
        g.append(vec_)
    g = np.array(g)
    
    M = np.zeros([len_th, len_th])
    for i in range(len_th):
        for j in range(len_th):
            M[i, j] = np.real(
                bra_ket(g[i], g[j])\
              - bra_ket(g[i], ansatz_(theta) @ conj_tp(ansatz_(theta)) @ g[j])
            )  

    state = H @ ansatz_(theta)
    C = []
    for i in range(len(theta)):
        C.append(bra_ket(g[i], state))
    C = np.array(C)   

    MC = np.linalg.pinv(M) @ C    
    return np.real(MC)

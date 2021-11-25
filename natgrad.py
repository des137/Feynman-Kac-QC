from utils import bra_ket

def nat_grad(theta, H, ansatz_, h=0.001):
#     np.set_printoptions(precision=17)
    theta = np.array(theta)
    len_th = len(theta)
    
    g = []
    for i in range(len_th):
        diff_ = np.zeros(len_th)
        diff_[i] = h
        vec_ = (ansatz_(theta + diff_) - ansatz_(theta))/h    
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
    for i in range(len_th):
        C.append(g[i], state)
    C = np.array(C)   

    MC = np.linalg.pinv(M) @ C    
    return np.real(MC)    

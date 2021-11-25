from utils import tensorprod, conj_tp, bra_ket, expectation
from gates import cx, rx, ry, rz, cry
from natgrad import nat_grad

# Define the computational basis state
Zero = np.array([1, 0]).reshape(-1, 1)
One = np.array([0, 1]).reshape(-1, 1)   

def evolve(initial_thetas, dt, TIMESTEPS, HAM, Ansatz):
    t_val = np.zeros([TIMESTEPS, len(initial_thetas)])
    t_val[0] = initial_thetas
    for i in range(1, TIMESTEPS):
        natgrad_ = nat_grad(t_val[i - 1], HAM, Ansatz).T
        t_val[i] = t_val[i - 1] - dt * natgrad_
    return t_val    

if __name__ == '__main__'    
    # Ansatz
    def Ansatz(theta):
        theta = np.array(theta)    
        init  = tensorprod(Zero, Zero)
        l_1 = tensorprod(rx(theta[0]), I_)
        l_2 = cry(theta[1])
        return l_2 @ l_1 @ init

    # Hamiltonian
    HAM = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0], 
        [0, 0, 3, 0], 
        [0, 0, 0, 0]
    ])
    
    # Evolution parameters
    t_i = np.array([pi/2, 0.05]) 
    dt =  0.1
    TIMESTEPS = 1000

    # Evolve
    evolve(t_i, dt, TIMESTEPS, HAM, Ansatz)

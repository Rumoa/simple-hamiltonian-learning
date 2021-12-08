#we are going to create a classical hamiltonian learning
#we have the following:
# P(x) which is our knowledge about the parameters
# P(data|H(x)) which is computing the born rule of a certain experiment given the parameter
# update it 


import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1], [1, 0]])*1j
Z = np.array([[1, 0], [0, -1]])

hbar = 1


def H_matrix(alpha):
    return alpha*X

def H(v, alpha):
    v = v.copy()
    if v.shape!=(0, 1):
        v = v.reshape(-1, 1)
    return np.matmul(H_matrix(alpha), v)


def evol_operator(H, t):
    return expm(-1j*t*H)

# evol_operator = np.vectorize(evol_operator)

def evol_vector(H, v, t):
    if hasattr(t, "__len__") is False:
        return np.matmul(evol_operator(H, t), v)
    else:
        result = np.zeros((len(t), 2, 1))
        for i, t_now in enumerate(t):
            result[i, :, :] = np.matmul(evol_operator(H, t_now), v)
        return result       



# evol_vector = np.vectorize(evol_vector)

alpha = 1
v = np.array([1, 0]).reshape(-1, 1)
t = np.linspace(0, 10, 50)
t_0 = 0
t_f = 0.4

evolved_arr = evol_vector(H_matrix(alpha), v, t)
plt.plot(t, evolved_arr[:, 0])
plt.show()
print(H)
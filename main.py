# we are going to create a classical hamiltonian learning
# we have the following:
# P(x) which is our knowledge about the parameters
# P(data|H(x)) which is computing the born rule of a certain experiment given the parameter
# update it


import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import random


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1], [1, 0]])*1j
Z = np.array([[1, 0], [0, -1]])

hbar = 1


def H_matrix(alpha):
    return alpha*X


def H(v, alpha):
    v = v.copy()
    if v.shape != (0, 1):
        v = v.reshape(-1, 1)
    return H_matrix(alpha)@v


def evol_operator(H, t):
    return expm(-1j*t*H)

# evol_operator = np.vectorize(evol_operator)


def evol_vectorize(H, v, t):
    result = np.zeros((len(t), 2, 1), dtype='complex_')
    for i, t_now in enumerate(t):
        result[i, :, :] = evol_operator(H, t_now)@v
    return result


def evol_state(H, v, t):
    if hasattr(t, "__len__") is False:
        return np.matmul(evol_operator(H, t), v)
    else:
        return evol_vectorize(H, v, t)


def prob(omega, t, d):
    return np.power(np.sin((omega*t/2)), 2*(1-d))*np.power(np.cos((omega*t/2)), 2*(d))


def sample_results(n, omega)    
    results = np.random.choice([0.0, 1.0], p=[prob(omega, np.random.uniform(t_0, t_f), 0), 1 - prob(omega, np.random.uniform(t_0, t_f), 0)],
                               size=n)
    return results


no_parameters = 5
prior = np.ones(no_parameters)
weights = np.ones(no_parameters)
# evol_vector = np.vectorize(evol_vector)

alpha = np.linspace(0, 1, no_parameters)
v = np.array([0, 1]).reshape(-1, 1)
measure_qubits = np.array([1, 0]).reshape(-1, 1)
t = np.linspace(0, 10, 100)
t_0 = 0.0
t_f = 4.0

evolution = np.zeros((len(alpha), len(t), 2, 1), dtype='complex_')
# the first component is each alpha of the hamiltonian
for i, alpha_i in enumerate(alpha):
    evolution[i, :, :, :] = evol_state(H_matrix(alpha_i), v, t)


# prob_0 = np.squeeze(np.array([np.conjugate(v.T)@i for i in evolved_arr[:, :, :]]))
prob_0 = np.power(np.abs(np.squeeze(
    np.array([np.conjugate(measure_qubits.T)@i for i in evolution]))), 2)
# plt.plot(t, np.real(prob_0[1, :]))


for i in range(len(alpha)):
    plt.plot(t, np.real(prob_0[i, :]), label='l')
    plt.legend(str(alpha[i]))
plt.show()

print("hello")

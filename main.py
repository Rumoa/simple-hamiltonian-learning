# we are going to create a classical hamiltonian learning
# we have the following:
# P(x) which is our knowledge about the parameters
# P(data|H(x)) which is computing the born rule of a certain experiment given the parameter
# update it
import math

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

import random


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1], [1, 0]])*1j
Z = np.array([[1, 0], [0, -1]])

hbar = 1


def H_matrix(alpha):
    return alpha/2*X


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
    return np.power(np.sin((omega*t/2)), 2*(d))*np.power(np.cos((omega*t/2)), 2*(1-d))

# def prob_0(omega, t):
#     return np.power(np.cos(t*omega/2),2)

def sample_results(n, omega):
    results = []
    for _ in range(n):
        t = np.random.uniform(t_0, t_f)
        p0 = prob(omega, t, 0)
        results.append(np.random.choice([0.0, 1.0], p=[p0, 1 - p0]))
    return results


no_parameters = 100
prior = np.ones(no_parameters)
weights = np.ones(no_parameters)
# evol_vector = np.vectorize(evol_vector)

alpha = np.linspace(0, 1, no_parameters)
v = np.array([1, 0]).reshape(-1, 1)/np.sqrt(1)
measure_qubits = np.array([1, 0]).reshape(-1, 1)
t = np.linspace(0, 10, 100)
t_0 = 0.0
t_f = 5.0
n_samples = 1000

evolution = np.zeros((len(alpha), len(t), 2, 1), dtype='complex_')
# the first component is each alpha of the hamiltonian
for i, alpha_i in enumerate(alpha):
    evolution[i, :, :, :] = evol_state(H_matrix(alpha_i), v, t)


# prob_0 = np.squeeze(np.array([np.conjugate(v.T)@i for i in evolved_arr[:, :, :]]))
prob_0 = np.power(np.abs(np.squeeze(
    np.array([np.conjugate(measure_qubits.T)@i for i in evolution]))), 2)
# plt.plot(t, np.real(prob_0[1, :]))
# plt.show()


# for i in range(len(alpha)):
#     plt.plot(t, np.real(prob_0[i, :]), label='l')
#     plt.legend(str(alpha[i]))
# plt.show()
# # samples = sample_results(50000, 0.8)
# print("hello")
#aqui aqui
alpha = [0, 1]
np.random.seed(1)

no_parameters = 50
no_samples = 50000
alpha = np.linspace(0, 1, no_parameters)
samples = sample_results(no_samples, 0.8)
probs = []
alpha = [0.0, 0.8, 1]
no_parameters=3
for i, alpha_i in enumerate(alpha):
    prob_parameters = []
    for i_sample, sample in enumerate(samples):
        t = 1/np.sqrt(np.power(np.random.choice(alpha)-np.random.choice(alpha), 2))
        while math.isinf(t):
            print("ojo")
            t = 1/np.sqrt(np.power(np.random.choice(alpha)-np.random.choice(alpha), 2))
        # print(t)
        evo = evol_state(H_matrix(alpha_i), v, t)
        prob_0 = np.power(
            np.abs(np.conjugate(measure_qubits.T)@evo), 2).ravel()
        # print(prob_0[0])
        prob_1 = 1 - prob_0
        if sample == 0.0:
            prob_parameters.append(prob_0[0])
        else:
            prob_parameters.append(prob_1[0])
    probs.append(prob_parameters)
probs = np.array(probs)
log_prob = np.log(probs)
log_likelihoods = np.sum(probs, axis=1)/no_samples
print(log_likelihoods)
# probs = np.array(probs).transpose().reshape(no_parameters, no_samples)
# print(probs)
print("hello")

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import joblib
from numba import jit


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1], [1, 0]])*1j
Z = np.array([[1, 0], [0, -1]])

h_bar = 1
np.random.seed(1)

def H(matrix, omega):
    return omega * matrix


@jit(nopython=True)
def mat_exp(A):
    d, Y = np.linalg.eig(A)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))

    B = Y@D@Yinv
    return B


def evolve_state(H, v, t, expfun="scipy"):
    if expfun=="scipy":
        evolved_state =  expm(-t*H*1j)@v
    else:
        evolved_state =  mat_exp(-t*H*1j)@v    
    return evolved_state/np.linalg.norm(evolved_state)

def prob_0(vec, tol=1E-8):
    pr = np.power(np.abs(vec[0]), 2)
    if np.abs(pr-1.0)<tol:
        pr = 1
    if np.abs(pr-0)<tol:
        pr = 0    
    return pr

def norm_H(H):
    return np.sqrt(np.max(np.linalg.eig(np.transpose(np.conjugate(H))@H)[0]))

def normalize_distribution(p):
    if p.sum() != 1.0:
        p = p*(1./p.sum())
    return p

def Sample(vec0, H, t0=0, tf=10, t_type="random", size = 100, t_step=0.1, t_single = 0):
    if t_type=="random":
        tgrid = np.arange(t0, tf, t_step)
        t = np.random.choice(tgrid, size= size)
    if t_type=="single":
        t = t_single*np.ones(size)
    probs = [prob_0(evolve_state(H, vec0, ti)) for ti in t ]
    return np.array([np.random.choice([0, 1], p=[pi, 1-pi]) for pi in probs])
    
state = np.array([1, 0])
state = state/np.linalg.norm(state)



# print(prob_0(state))
# print(prob_0(evolve_state(h, state, 1)))



# print(h)
# print(norm_H(h))




part_min = 0
part_max = 10

alpha = 4.2#0.834
h = H(X, alpha)

no_particles = 100
weights = normalize_distribution(np.ones(no_particles))
particles = np.arange(part_max/no_particles,part_max+part_max/no_particles,
                   part_max/no_particles)  #np.linspace(part_min, part_max, no_particles)

no_samples = 60

samples = Sample(state, h, 0, 0.5*3.1415, size = no_samples)
joblib.dump(samples, "samples/s1.job")
# samples = joblib.load("samples/s1.job")

# import sys
# sys.exit("Error message")

def PGH(particles, distribution):
    x1, x2 = np.random.choice(particles, size=2, p=normalize_distribution(distribution), replace=True)
    t = 1/ np.abs(x1 - x2 )
    return t

def Cov( particles, distribution):
    p = normalize_distribution(distribution)
    mu = np.dot(p, particles)
    sigma = np.dot(p,particles**2) - np.dot(mu, mu)
    return sigma


def resample(particles, distribution, a):
    prob = normalize_distribution(distribution)
    mu = np.average(particles, weights=prob)
    h = np.sqrt(1-a**2)
    Sigma = h**2 * Cov(particles, prob)
    new_weights = []
    new_particles = []
    for _ in range(len(particles)):
        part_candidate = np.random.choice(particles, size = 1, p=prob, replace=True)
        mu_i = a*part_candidate + (1-a)*mu
        part_prime = np.random.normal(mu_i, Sigma)
        new_particles.append(part_prime[0])
        new_weights.append(1/len(particles))

    return (np.array(new_particles), np.array(new_weights))

likelihoods = np.empty([no_samples, no_particles])

steps = 1000000




for i_step in range(steps):
    print("Sample no. ", i_step)
    # t = PGH(particles, weights)
    t = 1/np.sqrt(Cov(particles, weights))

    sample = Sample(state, h, t_type="single", size = 1, t_single=t)[0]
    print("time:", t)
    # if t>1E15:
    #     break
    print(np.average(particles, weights=normalize_distribution(weights)), np.var(particles))
    if np.var(particles)<1E-5:
        break
    probs_0 = [prob_0(evolve_state(H(X, particle_i), state, t))  for   particle_i in particles]   
    probs_sample = np.array([p0 if sample==0 else 1 - p0 for p0 in probs_0])
    # likelihoods[i_sample, :] = probs_sample
    new_weights = weights* probs_sample
    weights = normalize_distribution(new_weights)
    print("1/w^2: ", 1/np.sum(weights**2) )

    if 1/np.sum(weights**2) < no_particles/2:
        print("RESAMPLING")
        particles, weights = resample(particles, weights, a=0.98)
    # else:
    #     particles = np.random.choice(particles, size = no_particles, p=weights, replace=False)


estimated_parameter = np.sum(np.dot(weights, particles))
print(estimated_parameter)
# plt.plot(particles, weights)
# plt.show()
# t = np.linspace(start=0, stop=20, num = 200)
# evolution_prob = [prob_0(evolve_state(h, state, ti)) for ti in t ]
# plt.plot(t, evolution_prob)
# plt.ylim([0, 1.1])
# plt.yticks(np.arange(0, 1.1, step=0.1))

# plt.show()

# sample = sample(state, h, 0, 20, size = 10000)
# print(sample)
# joblib.dump(sample, "samples/s1.job")
print("end")
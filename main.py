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
rng = np.random.default_rng()

def H(free_model, *omega):
    return free_model(*omega)


def free_model(omega):
    return omega*X


def free_model_2(omega):
    return omega[0]*X + omega[1]*Y


@jit(nopython=True)
def mat_exp(A):
    d, Y = np.linalg.eig(A)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))

    B = Y@D@Yinv
    return B


def evolve_state(H, v, t):
    evolved_state = expm(-t*H*1j)@v
    return evolved_state/np.linalg.norm(evolved_state)


@jit(nopython=True)
def evolve_state_fast(H, v, t):
    evolved_state = mat_exp(-t*H*1j)@v
    return evolved_state/np.linalg.norm(evolved_state)


def prob_0(vec, tol=1E-8):
    pr = np.power(np.abs(vec[0]), 2)
    if np.abs(pr-1.0) < tol:
        pr = 1
    if np.abs(pr-0) < tol:
        pr = 0
    return pr


def norm_H(H):
    return np.sqrt(np.max(np.linalg.eig(np.transpose(np.conjugate(H))@H)[0]))


def normalize_distribution(p):
    '''
    Normalize probability distribution p. If multidimensional, it assumes that each row (axis=1) contains the entire p
    Each column will correspond to a different distribution

    Args:
        p: numpy array containing the probability distribution(s)

    Returns:
        The normalized probability function.
    '''
    if len(p.shape) == 1:
        return p/p.sum()
    else:
        return p/p.sum(axis=1).reshape(-1, 1)


# def Sample(vec0, H, t0=0, tf=10, t_type="random", size=100, t_step=0.1, t_single=0):
#     if t_type == "random":
#         tgrid = np.arange(t0, tf, t_step)
#         t = np.random.choice(tgrid, size=size)
#     if t_type == "single":
#         t = t_single*np.ones(size)
#     probs = [prob_0(evolve_state_fast(H, vec0, ti)) for ti in t]
#     return np.array([np.random.choice([0, 1], p=[pi, 1-pi]) for pi in probs])


def PGH(particles, distribution):
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        x1, x2 = np.random.choice(
            particles, size=2, p=normalize_distribution(distribution), replace=True)
        t = 1 / np.linalg.norm(x1-x2)
        return t
    else:
        M = particles.shape[1] #no of particles
        l1, l2 = np.random.choice(
                   M , size=2, p=normalize_distribution(distribution), replace=False)
        p1 = particles[:, l1]
        p2 = particles[:, l2]

        t = 1 / np.linalg.norm(p1 - p2)
        return t


def Mean(particles, distribution):
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        p = normalize_distribution(distribution)
        return (p*particles).sum()
    else:
        return (particles*normalize_distribution(distribution)).sum(axis=1)




def Cov(particles, distribution):
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        p = normalize_distribution(distribution)
        mu = Mean(particles, p)
        sigma = (p*(particles**2)).sum() - (mu*mu)
        return sigma
    else:
        p = normalize_distribution(distribution)
        mu = Mean(particles, p).reshape(-1, 1)
        D = particles.shape[0]
        cov_sum = np.zeros([D, D])
        for i in range(particles.shape[1]):
            part = particles[:, i].reshape(-1, 1)
            cov_sum = cov_sum + p[i]*part@part.T
            
        mu = Mean(particles, p).reshape(-1, 1)
        return cov_sum - mu@mu.T
        






def resample(particles, distribution, a):
    prob = normalize_distribution(distribution)
    mu = Mean(particles, prob)
    h = np.sqrt(1-a**2)
    Sigma = h**2 * np.sqrt(Cov(particles, prob))
    new_weights = []
    new_particles = []
    if len(particles.shape) == 1 and len(distribution.shape) == 1:
        for _ in range(len(particles)):
            part_candidate = np.random.choice(
                particles, size=1, p=prob, replace=False)
            mu_i = a*part_candidate + (1-a)*mu
            part_prime = np.random.normal(mu_i, Sigma)
            new_particles.append(part_prime[0])
            new_weights.append(1/len(particles))

        return (np.array(new_particles), np.array(new_weights))
    else:
        new_particles = np.zeros(particles.shape)
        new_weights = np.zeros(distribution.shape)
        M = particles.shape[1] #number of particles
        for i in range(particles.shape[0]):
            for j in range(particles.shape[1]):
                loc_candidate = np.random.choice(
                    M , size=1, p=prob, replace=False)
                part_candidate = particles[i, loc_candidate]
                # print(part_candidate)
                mu=Mean(particles[i, :], prob)
                Sigma = h**2 * np.sqrt(Cov(particles[i, :], prob))
                mu_i = a*part_candidate + (1-a)*mu
                part_prime = np.random.normal(mu_i, Sigma)
                new_particles[i, j] = part_prime
                new_weights[j] = 1/M
   
        return (new_particles, new_weights)



def MSE(x, xtrue):
    return np.power(x - xtrue, 2)


def Prob_sample(t,  omega, T2):
    return np.exp(-t/T2)*np.power(np.cos(0.5*omega*t),2) + (1 -np.exp(-t/T2))*0.5

def Sample(t, omega, T2):
    p1 = Prob_sample(t, omega, T2)
    # result = np.random.choice([0, 1], p=[p0, p1])
    result = rng.binomial(1, p1)
    
    return result


def update_SMC(t, particles, weights,  true_parameters):
    sample = Sample(t, true_parameters[0], true_parameters[1])

    if len(particles.shape) == 1 and len(weights.shape) == 1:
        probs_0 = [Prob_sample(t, particle_i[0], particle_i[1])
                    for particle_i in particles]
        
    else:
        probs_0 = [Prob_sample(t, particle_i[0], particle_i[1])
                    for particle_i in particles.T]
    probs_sample = np.array([p1 if sample == 1 else 1 - p1 for p1 in probs_0])
    # likelihoods[i_sample, :] = probs_sample
    new_weights = weights * probs_sample
    n_weights = normalize_distribution(new_weights)

    return particles, n_weights


def adaptive_bayesian_learn(particles, weights, true_parameters,  steps, tol=1E-5):
    for i_step in range(steps):
        # print("Sample no. ", i_step)
        t = PGH(particles, weights)
        # t = 1/np.sqrt(Cov(particles, weights))
        # t = 1/np.sqrt(np.max(Cov(particles, weights)))
        # t = 1/np.sqrt(np.linalg.det(Cov(particles, weights)))


        # print("time:", t)
        # print("Mean", Mean(particles, normalize_distribution(weights)))
        # print("Cov", Cov(particles, normalize_distribution(weights)) ) 


        particles, weights = update_SMC(
            t, particles, weights, true_parameters=true_parameters)
        # print("1/w^2: ", 1/np.sum(weights**2))

        if 1/np.sum(weights**2) < no_particles*(2/3):
            print("RESAMPLING")
            particles, weights = resample(particles, weights, a=0.9)

        # if np.max(Cov(particles, weights)) < tol:
            # break

        if i_step%25==0:
            estimated_parameter = Mean(particles, weights)
            print("MSE: ", Mse(estimated_parameter,true_parameters))
    # estimated_parameter = np.sum(np.dot(weights, particles))
    estimated_parameter = Mean(particles, weights)

    
    return estimated_parameter


state = np.array([1, 0], dtype=np.complex128)
state = state/np.linalg.norm(state)


def Mse(par, true_par):
    suma = 0
    for i in range(len(par)):
        suma = suma + (par[i] - true_par[i])**2
    return suma    


bounds = np.array([[0, 1],
                    [2, 3]])

# bounds = np.array([[0, 10]])
alpha1 =0.8# 0.834
alpha2 =2.6
# h = H(free_model_2, [alpha1, alpha2])
# h = H(free_model, alpha1)
D = 2 #dimension

true_parameters = np.array([alpha1, alpha2])
no_particles = 200
weights = normalize_distribution(np.ones(no_particles))

# weights = normalize_distribution(np.normal())
particles = np.zeros([D, no_particles])

for i in range(bounds.shape[0]):
    p_min, p_max = bounds[i, :]
    # particles[i, :] = np.linspace(p_min, p_max, no_particles)
    particles[i, :] = np.random.uniform(p_min, p_max, no_particles)
    # particles[i, :] = ((p_min + p_max)/2) *np.random.normal(0,1 , no_particles) + (p_max-p_min)/2

if D==1:
    particles = particles[0, :]

steps = 1000
np.random.seed(1)
estimated_alpha = adaptive_bayesian_learn(
    particles=particles, weights=weights,  true_parameters=true_parameters, steps=steps,  tol=5E-3)
print("Estimated results: ", estimated_alpha)
# print(MSE(estimated_alpha, alpha))
print("Finish")

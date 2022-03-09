import numpy as np
from scipy.linalg import expm
from numba import jit

from timeit import default_timer as timer
 
A = np.array([[1, -4, -5], [2, 3, 10], [-6, 5, -10]], dtype=np.float32)
@jit(nopython=True)
def mat_exp(A):
    d, Y = np.linalg.eig(A)
    Yinv = np.linalg.pinv(Y)
    D = np.diag(np.exp(d))

    B = Y@D@Yinv
    return B

mat_exp(A)
start = timer()
mat_exp(A)
end = timer()
print(end - start  )

start = timer()
expm(A)
end = timer()
print(end - start )

import numpy as np
from numba import jit, prange

# Latent Matrix Factorization using SGD

# L is m x rank, R is rank x n
# Parallel SGD, alternating on L and R
@jit(parallel=True, nopython=True)
def do_epoch(train: np.ndarray, L: np.ndarray, R: np.ndarray, alpha: float, i: int):
    m, n = train.shape
    modify_L = (i % 2 == 0)
    if modify_L:
        for row in prange(m):
            for col in range(n):
                if train[row, col] != 0:
                    est = np.dot(L[row], R[:, col])
                    if est < 1 and train[row, col] == 1:
                        continue
                    elif est > 5 and train[row, col] == 5:
                        continue
                    resid = train[row, col] - est
                    L[row] += alpha * resid * R[:, col]
    else:
        for col in prange(n):
            for row in range(m):
                if train[row, col] != 0:
                    est = np.dot(L[row], R[:, col])
                    if est < 1 and train[row, col] == 1:
                        continue
                    elif est > 5 and train[row, col] == 5:
                        continue
                    resid = train[row, col] - est
                    R[:, col] += alpha * resid * L[row]

# Makes CV
@jit(parallel=True, nopython=True)
def makeCrossVal(train: np.ndarray, k: int):
    output = np.zeros((train.shape[0], train.shape[1], k))
    idxs = np.random.permutation(np.arange(0, train.shape[1], 1))

    for i in prange(k):
        for row in prange(train.shape[0]):
            idxStart = ((row + i) % k) * (len(idxs) // k)
            idxEnd = ((row + i + 1) % k) * (len(idxs) // k)
    
            if (row + i + 1) % k == k - 1:
                idxEnd = len(idxs)

            for j in range(idxStart, idxEnd):
                idx = idxs[j]
                output[row, idx, i] = train[row, idx]
    
    return output


def get_train_error(_train, _L, _R):
    est = _L @ _R
    est = np.clip(est, a_min=1, a_max=5)
    idxs = np.where(_train > 0)
    error = np.square(_train[idxs] - est[idxs]).mean()
    return np.sqrt(error)
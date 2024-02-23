import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import statsmodels.api as sm

# Some EDA I did on the time data to see if there were any simple patterns, not used in final project.

@jit(parallel=True, nopython=True)
def getQuantiles(time):
    toRet = np.zeros(time.shape[1])
    for col in prange(time.shape[1]):
        movieTimes = time[:, col]
        movieTimes = movieTimes[movieTimes > 0]
        max = np.max(movieTimes)
        if max == 0:
            toRet[col] = 0
            continue
        
        toRet[col] = np.quantile(movieTimes, 0.6) - np.quantile(movieTimes, 0.4)
    
    return toRet

@jit(parallel=True, nopython=True)
def getScoreSum(ratings):
    toRet = np.zeros(ratings.shape[1])
    for col in prange(ratings.shape[1]):
        for row in range(ratings.shape[0]):
            if ratings[row, col] == 0:
                continue
            else:
                toRet[col] += ratings[row, col]
                
    return toRet


ratings_sparse = np.load("trainMatrix.npy")
time_sparse = np.load("timeTrain.npy")

timeRanges = getQuantiles(time_sparse)
meanScores = np.divide(getScoreSum(ratings_sparse), (ratings_sparse > 0).sum(axis=0))
# print(meanScores)



gt0 = timeRanges > 0
realTimeRanges = timeRanges[gt0]
realMeanScores = meanScores[gt0]

x1 = np.log(realTimeRanges).reshape(-1, 1)
# x2 = realTimeRanges.reshape(-1, 1)
regX = sm.add_constant(x1)

reg = sm.OLS(realMeanScores, regX)
fit = reg.fit()

print(fit.summary())

plt.scatter(timeRanges, meanScores, alpha=0.1)
plt.show()




# plt.hist(meanScores)
# plt.show()

# plt.hist(timeRanges)
# plt.yscale("log")
# plt.show()
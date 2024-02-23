import numpy as np
import matplotlib.pyplot as plt
from matrixLFM import makeCrossVal, do_epoch, get_train_error

# Figures out how many epochs we need on each rank to minimize the CV error

train = np.load("trainMatrix.npy")
test = np.load("testMatrix.npy")

np.random.seed(423)

minrank = 5
ranks = np.arange(minrank, 20, 1)
x = 2000
k = 10
folds = makeCrossVal(train, k)

foldSet = set([x for x in range(k)])
foldTestError = {}

for foldNum in range(k):
    for rank in ranks:
        foldTest = folds[:, :, foldNum]
        fold = np.zeros(train.shape)
        L = np.random.rand(train.shape[0], rank)
        R = np.random.rand(rank, train.shape[1])

        foldTestError[(foldNum, rank)] = []

        for j in foldSet.difference([foldNum]):
            fold += folds[:, :, j]
        
        for i in range(x):
            # Decrease the learning rate every 10 iterations
            lr = 0.15 * (1 / ((i // 10) + 1))
            
            do_epoch(fold, L, R, lr, False)
            if (i % 50 == 0) and (i != 0):
                cvError = get_train_error(foldTest, L, R)
                foldTestError[(foldNum, rank)].append((i, cvError))
                print(f"Fold {foldNum}, Rank {rank}: Train error: {get_train_error(fold, L, R)}, Test Error: {cvError}")
        print()


errors = np.zeros((len(foldTestError[(0, minrank)]), len(ranks), k))
idxs = None
for rank in ranks:
    for foldNum in range(k):
        idxs, error = zip(*foldTestError[(foldNum, rank)])
        errors[:, rank - minrank, foldNum] = error
        # plt.plot(idxs, error)

# plt.show()

meanError = np.mean(errors, axis=2)

for rank in ranks:
    plt.plot(idxs, meanError[:, rank - minrank], label=f"Rank {rank}")
plt.title("Mean CV Error: Rank")
plt.xlabel("Epoch #")
plt.ylabel("RMSE")
plt.legend()
plt.show()



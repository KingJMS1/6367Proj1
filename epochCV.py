import numpy as np
import matplotlib.pyplot as plt
from matrixLFM import makeCrossVal, do_epoch, get_train_error

# Figures out how many epochs we need to minimize the CV error, not used in final paper

train = np.load("trainMatrix.npy")
test = np.load("testMatrix.npy")

rank = 9
x = 3000
k = 10
folds = makeCrossVal(train, k)

foldSet = set([x for x in range(k)])
foldTestError = {}

for foldNum in range(k):
    foldTest = folds[:, :, foldNum]
    fold = np.zeros(train.shape)
    L = np.random.rand(train.shape[0], rank)
    R = np.random.rand(rank, train.shape[1])

    foldTestError[foldNum] = []

    for j in foldSet.difference([foldNum]):
        fold += folds[:, :, j]
    
    for i in range(x):
        # Decrease the learning rate every 50 iterations
        lr = 0.2 * (1 / ((i // 50) + 1))
        
        do_epoch(fold, L, R, lr, False)
        if (i % 200 == 0):
            cvError = get_train_error(foldTest, L, R)
            foldTestError[foldNum].append((i, cvError))
            print(f"Fold {foldNum}: Train error: {get_train_error(fold, L, R)}, Test Error: {cvError}")
    print()


errors = np.zeros((len(foldTestError[0]), k))
idxs = None
for foldNum in range(k):
    idxs, error = zip(*foldTestError[foldNum])
    errors[:, foldNum] = error
    plt.plot(idxs, error)

plt.show()

meanError = np.mean(errors, axis=1)

plt.plot(idxs, error)
plt.title("Mean CV Error: Epoch #")
plt.xlabel("Epoch #")
plt.ylabel("RMSE")
plt.show()



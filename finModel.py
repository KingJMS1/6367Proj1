import numpy as np
import matplotlib.pyplot as plt
from matrixLFM import makeCrossVal, do_epoch, get_train_error
import seaborn as sns
import pandas as pd

# Trains the final model based on CV results and makes some graphs

train = np.load("trainMatrix.npy")
test = np.load("testMatrix.npy")

rank = 13
x = 600

L = np.random.rand(train.shape[0], rank)
R = np.random.rand(rank, train.shape[1])
    
for i in range(x):
    # Decrease the learning rate every 10 iterations
    lr = 0.15 * (1 / ((i // 10) + 1))
    
    do_epoch(train, L, R, lr, False)

error = get_train_error(test, L, R)

preds = L @ R
clipPreds = np.clip(preds, a_max=5, a_min=1)
sns.heatmap(clipPreds[:20, :20])
plt.title("Predictions")
plt.xlabel("Moive ID - 1")
plt.ylabel("User ID - 1")
plt.show()
plt.clf()

sns.heatmap(train[:20, :20])
plt.title("Training")
plt.xlabel("Moive ID - 1")
plt.ylabel("User ID - 1")
plt.show()
plt.clf()

sns.heatmap(test[:20, :20])
plt.title("Testing")
plt.xlabel("Movie ID - 1")
plt.ylabel("User ID - 1")
plt.show()

sns.heatmap((train - clipPreds)[:20, :20] - 20 * (train == 0)[:20, :20], vmin=-7, vmax=4, center=0)
plt.title("Training - Predicted")
plt.xlabel("Movie ID - 1")
plt.ylabel("User ID - 1")
plt.show()


sns.heatmap((test - clipPreds)[:20, :20] - 20 * (test == 0)[:20, :20], vmin=-7, vmax=4, center=0)
plt.title("Testing - Predicted")
plt.xlabel("Movie ID - 1")
plt.ylabel("User ID - 1")
plt.show()



idxs = np.where(test > 0)
# idxs = np.where(train > 0)
preds = preds[idxs]
reals = test[idxs]
# reals = train[idxs]

print(get_train_error(test, L, R))

plt.clf()
plt.scatter(np.clip(preds, a_min=1, a_max=5) + np.random.normal(0, 0.05, len(preds)), reals + np.random.normal(0, 0.1, len(preds)), alpha=0.01)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Predicted vs Actual")
plt.show()

plt.clf()

sns.boxplot(L[:20, :].T)
plt.title("User factors (first 20 users)")
plt.xlabel("User #")
plt.ylabel("Value (Effect amount)")
plt.show()

plt.clf()

sns.boxplot(R[:, :20])
plt.title("Movie factors")
plt.xlabel("Movie #")
plt.ylabel("Value (Effect amount)")
plt.show()
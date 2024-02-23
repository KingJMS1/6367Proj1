import pandas as pd
import numpy as np

# Loads the data into a useful format.

_train = pd.read_table("movielens_100k.base", names=["userID", "movieID", "rating", "timestamp"])
_test = pd.read_table("movielens_100k.test", names=["userID", "movieID", "rating", "timestamp"])

m = _train["userID"].max()
n = _train["movieID"].max()
trainArr = np.zeros((m, n))
testArr = np.zeros((m, n))
timestamps_train = np.zeros((m, n))
timestamps_test = np.zeros((m, n))

for item in _train.iterrows():
    item = item[1]
    trainArr[item["userID"] - 1, item["movieID"] - 1] = item["rating"]
    timestamps_train[item["userID"] - 1, item["movieID"] - 1] = item["timestamp"]

for item in _test.iterrows():
    item = item[1]
    testArr[item["userID"] - 1, item["movieID"] - 1] = item["rating"]
    timestamps_test[item["userID"] - 1, item["movieID"] - 1] = item["timestamp"]

np.save("trainMatrix", trainArr)
np.save("testMatrix", testArr)
np.save("timeTrain", timestamps_train)
np.save("timeTest", timestamps_test)

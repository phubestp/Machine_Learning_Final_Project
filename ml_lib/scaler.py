import numpy as np

class MinMaxScaler:

    def __init__(self) -> None:
        pass

    def scale(self, X):
        new_X = X
        for i in range(X.shape[1]):
            min_x = min(X[:, i])
            max_x = max(X[:, i])
            for j in range(X.shape[0]):
                new_X[j][i] = (X[j][i] - min_x) / (max_x - min_x)
        return new_X
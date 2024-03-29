import numpy as np

class SVM:

    def __init__(self, d) -> None:
        self.d = d
        self.w = np.zeros(d+1)
    
    def fit(self, X_train, y_train):
        while True:
            m = 0
            for d in zip(X_train, y_train):
                x = np.array([*d[0], 1]).reshape(-1, 1)
                y = d[1]
                if y * np.dot(self.w, x) <= 0:
                    self.w = self.w + y*np.array([*d[0], 1]).reshape(1, -1)
                    m = m + 1
            if m == 0:
                break
        print(self.w)
    
    def predict(self, x):
        return 1 if np.dot(self.w, np.array([*x, 1]).reshape(-1, 1)) > 0 else -1

    def score(self, x_test, y_test):
        correct = 0
        for d in zip(x_test, y_test):
            print(self.predict(d[0])
        return corr
        
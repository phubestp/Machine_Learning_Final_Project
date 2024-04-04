import numpy as np

class Perceptron:

    def __init__(self, d) -> None:
        self.d = d  
        self.w = np.zeros(d+1) 

    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train

    def create_hyperplane(self):
        while True:
            m = 0
            for d in zip(self.X_train, self.y_train):
                x = np.array([*d[0], 1]).reshape(-1, 1)
                y = d[1]
                if y * np.dot(self.w, x) <= 0:
                    self.w = self.w + y*np.array([*d[0], 1]).reshape(1, -1)
                    m = m + 1
            if m == 0:
                break
        print(self.w)

    def predict(self, X_test):
        self.create_hyperplane()
        predicted = []
        for x in X_test:
            if np.dot(self.w.T, np.array([*x, 1])) > 0:
                predicted.append(1)
            else:
                predicted.append(-1)
        return predicted

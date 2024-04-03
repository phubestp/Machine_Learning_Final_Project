import numpy as np
import math

class SVM:

    def __init__(self, d, C=1, iteration=1000, learning_rate=0.001) -> None:
        self.d = d
        self.C = C
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.w = np.zeros(d)
        self.b = 0
    
    # def hinge_loss(self):
    #     return np.dot(self.w, self.w.T) + self.C * sum([max(1 - y * (np.dot(self.w, x.T) + self.b), 0) for (x, y) in zip(self.X_train, self.y_train)]) 
    
    # def rbf_kernel(self, X, Z, gamma=1):
    #     distance_sq = np.sum((X[:, np.newaxis] - Z) ** 2, axis=2)
    #     return np.exp(-gamma * distance_sq)

    # def poly_kernel(self, X, Z, degree=3):
    #     return (np.dot(X, Z.T) + 1)**degree

    def calc_grad(self):
        dw = np.zeros(self.d)
        db = 0
        for (x, y) in zip(self.X_train, self.y_train): 
            if y * (np.dot(self.w, x.T) + self.b) < 1:
                dw += self.w - (self.C * y * x)
                db += -1 * self.C * y
            else:
                dw += self.w
        return dw, db
    
    def grad_desc(self):
        for i in range(self.iteration):
            dw, db = self.calc_grad()
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        self.grad_desc()
        predicted = []
        for x in X_test:
            if -1*np.dot(self.w, x) + self.b > 0:
                predicted.append(-1)
            else:
                predicted.append(1)
        return predicted


        
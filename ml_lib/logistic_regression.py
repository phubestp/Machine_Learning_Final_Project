import math
import numpy as np

class LogisticRegression:

    def __init__(self, d, iteration=1000, learning_rate=0.001) -> None:
        self.d = d
        self.w = np.zeros(d)
        self.b = 0
        self.iteration = iteration
        self.learning_rate = learning_rate

    def calc_grad(self, y_train):
        dw = np.zeros(self.d)
        db = 0
        for (x, y) in zip(self.X_train, y_train):
            dw += x*y*(np.exp(y*np.dot(x.T, self.w) + self.b)) / ((np.exp(y*np.dot(x.T, self.w) + self.b) + 1))
            db += y*(np.exp(y*np.dot(x.T, self.w) + self.b)) / ((np.exp(y*np.dot(x.T, self.w) + self.b) + 1))
        return dw, db
    
    def grad_desc(self, y_train):
        self.w = np.zeros(self.d)
        self.b = 0
        for i in range(self.iteration):
            dw, db = self.calc_grad(y_train)
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, y_train=None):
        if len(set(self.y_train)) == 2:
            self.grad_desc(self.y_train)
        else:
            self.grad_desc(y_train)
        predicted = []
        for x in X_test:
            pos = 1/(1+np.exp((1*((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            neg = 1/(1+np.exp((-1*((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            if pos > neg:
                predicted.append(1)
            else: 
                predicted.append(-1)
        return predicted
    
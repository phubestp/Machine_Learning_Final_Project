import numpy as np
import math

class LogisticRegression:

    def __init__(self, d, iteration=1000, learning_rate=0.001) -> None:
        self.d = d 
        self.w = np.zeros(d+1)  
        self.iteration = iteration  
        self.learning_rate = learning_rate 
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calc_grad(self):
        dw = np.zeros(self.d+1) 
        for (x, y) in zip(self.X_train, self.y_train):
            x = np.array([*x, 1])
            dw += -y*x*(np.exp(-y*np.dot(x.T, self.w))) / ((np.exp(-y*np.dot(x.T, self.w)) + 1))
        return dw
    
    def grad_desc(self):
        for i in range(self.iteration):
            # Calculate gradients
            dw = self.calc_grad()
            # Update weights and bias using gradient descent
            self.w -= self.learning_rate * dw

    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train 
    
    def predict(self, X_test):
        self.grad_desc()
        predicted = []  # Initialize list to store predictions
        for x in X_test:
            x = np.array([*x, 1])
            if self.sigmoid(np.dot(self.w.T, x)) > 0.5:
                predicted.append(1)
            else: 
                predicted.append(-1)
        return predicted  

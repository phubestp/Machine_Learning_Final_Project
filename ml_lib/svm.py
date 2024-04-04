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

    def calc_grad(self):
        dw = np.zeros(self.d) 
        db = 0  
        for (x, y) in zip(self.X_train, self.y_train): 
            # Check if the sample is not correctly classified
            if y * (np.dot(self.w.T, x) + self.b) < 1:
                dw += self.w - (self.C * y * x)  # Update gradients for misclassified sample
                db += -1 * self.C * y  # Update gradient of bias
            else:
                dw += self.w  # Update gradients for correctly classified sample
        return dw, db
    
    def grad_desc(self):
        for i in range(self.iteration):
            dw, db = self.calc_grad() 
            self.w -= self.learning_rate * dw  # Update weights
            self.b -= self.learning_rate * db  # Update bias

    def fit(self, X_train, y_train):
        self.X_train = X_train 
        self.y_train = y_train 

    def predict(self, X_test):
        self.grad_desc()  # Perform gradient descent to optimize parameters
        predicted = []
        for x in X_test:
            # Predict label based on decision boundary
            if np.dot(self.w.T, x) + self.b < 0:
                predicted.append(-1)
            else:
                predicted.append(1)
        return predicted

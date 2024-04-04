import numpy as np
import math

class LogisticRegression:

    def __init__(self, d, iteration=1000, learning_rate=0.001) -> None:
        """
        Initialize logistic regression model
        
        Parameters:
        - d: Dimensions of feature
        - iteration: Number of iterations for gradient descent
        - learning_rate: Learning rate for gradient descent 
        """
        self.d = d 
        self.w = np.zeros(d+1)  
        self.iteration = iteration  
        self.learning_rate = learning_rate 
    
    def sigmoid(self, z):
        """
        Calculate Sigmoid function
        
        Parameters:
        - z: value
        
        Returns:
        - Value from sigmoid function
        """
        return 1 / (1 + np.exp(-z))

    def calc_grad(self):
        """
        Calculate gradients of weights and bias

        Returns:
        - Gradients of weights and bias
        """
        dw = np.zeros(self.d+1) 
        for (x, y) in zip(self.X_train, self.y_train):
            x = np.array([*x, 1])
            dw += -y*x*(np.exp(-y*np.dot(x.T, self.w))) / ((np.exp(-y*np.dot(x.T, self.w)) + 1))
        return dw
    
    def grad_desc(self):
        """
        Perform gradient descent to update weights and bias

        """
        for i in range(self.iteration):
            # Calculate gradients
            dw = self.calc_grad()
            # Update weights and bias using gradient descent
            self.w -= self.learning_rate * dw

    def fit(self, X_train, y_train):
        """
        Fit the logistic regression model
        
        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        """
        self.X_train = X_train 
        self.y_train = y_train 
    
    def predict(self, X_test):
        """
        Predict labels for test data
        
        Parameters:
        - X_test: Test data features
        
        Returns:
        - Predicted labels for test data
        """
        self.grad_desc()
        predicted = []  # Initialize list to store predictions
        for x in X_test:
            x = np.array([*x, 1])
            if self.sigmoid(np.dot(self.w.T, x)) > 0.5:
                predicted.append(1)
            else: 
                predicted.append(-1)
        return predicted  

import math
import numpy as np

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
        self.w = np.zeros(d)  
        self.b = 0  
        self.iteration = iteration  
        self.learning_rate = learning_rate 

    def calc_grad(self, y_train):
        """
        Calculate gradients of weights and bias
        
        Parameters:
        - y_train: labels
        
        Returns:
        - Gradients of weights and bias
        """
        dw = np.zeros(self.d) 
        db = 0  
        for (x, y) in zip(self.X_train, y_train):
            dw += x * y * (np.exp(y * np.dot(x.T, self.w) + self.b)) / ((np.exp(y * np.dot(x.T, self.w) + self.b) + 1))
            db += y * (np.exp(y * np.dot(x.T, self.w) + self.b)) / ((np.exp(y * np.dot(x.T, self.w) + self.b) + 1))
        return dw, db
    
    def grad_desc(self, y_train):
        """
        Perform gradient descent to update weights and bias
        
        Parameters:
        - y_train: labels
        """
        self.w = np.zeros(self.d)  
        self.b = 0  
        for i in range(self.iteration):
            # Calculate gradients
            dw, db = self.calc_grad(y_train)
            # Update weights and bias using gradient descent
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def fit(self, X_train, y_train):
        """
        Fit the logistic regression model
        
        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        """
        self.X_train = X_train  # Store training features
        self.y_train = y_train  # Store training labels
    
    def predict(self, X_test):
        """
        Predict labels for test data
        
        Parameters:
        - X_test: Test data features
        
        Returns:
        - Predicted labels for test data
        """
        predicted = []  # Initialize list to store predictions
        for x in X_test:
            # Calculate probabilities using logistic function
            pos = 1 / (1 + np.exp((1 * ((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            neg = 1 / (1 + np.exp((-1 * ((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            # Make prediction based on higher probability
            if pos > neg:
                predicted.append(1)
            else: 
                predicted.append(-1)
        return predicted  

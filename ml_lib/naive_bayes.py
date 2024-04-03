import math

class GaussianNaiveBayes:

    def fit(self, X_train, y_train):
        """
        Fit the Gaussian Naive Bayes model.
        
        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        """
        self.X_train = X_train  
        self.y_train = y_train  
    
    def mean(self, X, y):
        """
        Calculate the mean of a feature for a given class.
        
        Parameters:
        - X: Features
        - y: Class label
        
        Returns:
        - Mean of the feature for the given class
        """
        summation = 0
        for i in range(len(X)):
            if self.y_train[i] == y:
                summation += X[i]
        return summation / len(X)
    
    def sd(self, X, y):
        """
        Calculate the standard deviation of a feature for a given class.
        
        Parameters:
        - X: Feature vector
        - y: Class label
        
        Returns:
        - Standard deviation of the feature for the given class
        """
        m = self.mean(X, y)
        summation = 0
        for i in range(len(X)):
            if self.y_train[i] == y:
                summation += (X[i] - m) ** 2
        return math.sqrt(summation / len(X))

    def gaussian(self, x, X, y):
        """
        Compute the probability function of a Gaussian distribution.
        
        Parameters:
        - x: Value at which to compute the probability 
        - X: Feature
        - y: Class label
        
        Returns:
        - Probability at the given value for the given class
        """
        sd = self.sd(X, y)
        mean = self.mean(X, y)
        return 1 / (math.sqrt(2 * math.pi * sd)) * math.e ** (-(1 / 2) * ((x - mean) / sd) ** 2)

    def prob(self, x, y):
        """
        Compute the probability of a data point belonging to a class.
        
        Parameters:
        - x: Data point
        - y: Class label
        
        Returns:
        - Probability of the data point belonging to the given class
        """
        p = 1
        for a in range(len(x)):
            p *= self.gaussian(x[a], self.X_train[:, a], y)
        return p
        
    def predict(self, new_x):
        """
        Predict the labels for new data points.
        
        Parameters:
        - new_x: New data points to predict labels for
        
        Returns:
        - Predicted labels for the new data points
        """
        predicted = []
        for x in new_x:
            if self.prob(x, 1) > self.prob(x, -1):
                predicted.append(1)
            else:
                predicted.append(-1)
        return predicted

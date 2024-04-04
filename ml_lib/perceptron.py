import numpy as np

class Perceptron:

    def __init__(self, d) -> None:
        """
        Initialize the Perceptron.
        
        Parameters:
        - d: Dimension of features
        """
        self.d = d  
        self.w = np.zeros(d+1) 

    def fit(self, X_train, y_train):
        """
        Fit the Perceptron model.
        
        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        """
        self.X_train = X_train 
        self.y_train = y_train

    def create_hyperplane(self):
        """
        Fit the Perceptron model.
        
        Parameters:
        - X_train: Training data features
        - y_train: Training data labels
        """
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
        """
        Predict labels for test data using the trained Perceptron model.
        
        Parameters:
        - X_test: Test data features
        
        Returns:
        - Predicted labels for the test data
        """
        self.create_hyperplane()
        predicted = []
        for x in X_test:
            # Predict label based on decision boundary
            if np.dot(self.w, np.array([*x, 1]).reshape(-1, 1)) > 0:
                predicted.append(1)
            else:
                predicted.append(-1)
        return predicted

import numpy as np
import statistics as st  # Importing necessary libraries

class KNN:
    
    def __init__(self, n_neighbors, p):
        """
        Initialize KNN
        
        Parameters:
        - n_neighbors: Number of neighbors
        - p: Power parameter for distance 
        """
        self.n_neighbors = n_neighbors  
        self.p = p  
        
    def fit(self, x_train, y_train):
        """
        Fit the KNN model
        
        Parameters:
        - x_train: Training data features
        - y_train: Training data labels
        """
        self.x_train = x_train  
        self.y_train = y_train 
        
    def predict(self, new_x):
        """
        Predict the labels for new data points
        
        Parameters:
        - new_x: New data points to predict labels for
        
        Returns:
        - Predicted labels for new data points
        """
        predict_list = []  # Initialize list to store predictions
        for new in new_x:
            # Calculate distances between new data point and all training data points
            prediction_list = [(self.distance(x, new), y) for x, y in zip(self.x_train, self.y_train)]
            # Convert prediction list to numpy array for sorting
            prediction_array = np.array(prediction_list, dtype=[('d', float), ('y', float)])
            prediction_array.sort(order='d')
            # Take the mode of the labels 
            predict_list.append(st.mode([x[1] for x in prediction_array[:self.n_neighbors]]))
        return np.array(predict_list)  
    
    def distance(self, p1, p2): 
        """
        Calculate the distance between two points
        
        Parameters:
        - p1: First point
        - p2: Second point
        
        Returns:
        - Minkowski distance between the two points
        """
        sum = 0  # Initialize sum for distance calculation
        for x_p1, x_p2 in zip(p1, p2):
            # Calculate sum of absolute differences raised to the power p
            sum += pow(abs(x_p1 - x_p2), self.p)
        return pow(sum, 1/self.p)  # Return the pth root of the sum of differences raised to power p

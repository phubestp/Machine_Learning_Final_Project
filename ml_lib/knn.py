import numpy as np
import statistics as st

class KNN:
    
    def __init__(self, n_neighbors, p):
        self.n_neighbors = n_neighbors
        self.p = p
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, new_x):
        predict_list = []
        for new in new_x:
            prediction_list = [(self.distance(x, new), y) for x, y in zip(self.x_train, self.y_train)]
            prediction_array = np.array(prediction_list, dtype=[('d', float), ('y', float)])
            prediction_array.sort(order='d')
            predict_list.append(st.mode([x[1] for x in prediction_array[:self.n_neighbors]]))
        return np.array(predict_list)
    
    def distance(self, p1, p2): 
        sum = 0
        for x_p1, x_p2 in zip(p1, p2):
            sum += pow(abs(x_p1 - x_p2), self.p)
        return pow(sum, 1/self.p)
    
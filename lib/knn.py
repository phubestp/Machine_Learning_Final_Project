class KNN:
    
    def __init__(self, n_neighbors, p):
        self.n_neighbors = n_neighbors
        self.p = p
        
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def predict(self, new_x):
        prediction_list = [(self.distance(x, new_x), y) for x, y in zip(self.x_train, self.y_train)]
        prediction_array = np.array(prediction_list, dtype=[('d', float), ('y', int)])
        prediction_array.sort(order='d')
        return st.mode([x[1] for x in prediction_array[:self.n_neighbors]])
        
    def distance(self, p1, p2): 
        sum = 0
        for x_p1, x_p2 in zip(p1, p2):
            sum += pow(abs(int(x_p1) - int(x_p2)), self.p)
        return pow(sum, 1/self.p)

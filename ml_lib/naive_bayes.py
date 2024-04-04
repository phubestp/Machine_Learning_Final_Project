import math

class GaussianNaiveBayes:

    def fit(self, X_train, y_train):
        self.X_train = X_train  
        self.y_train = y_train  
    
    def mean(self, X, y):
        summation = 0
        for i in range(len(X)):
            if self.y_train[i] == y:
                summation += X[i]
        return summation / len(X)
    
    def sd(self, X, y):
        m = self.mean(X, y)
        summation = 0
        for i in range(len(X)):
            if self.y_train[i] == y:
                summation += (X[i] - m) ** 2
        return math.sqrt(summation / len(X))

    def gaussian(self, x, X, y):
        sd = self.sd(X, y)
        mean = self.mean(X, y)
        return 1 / (math.sqrt(2 * math.pi * sd)) * math.e ** (-(1 / 2) * ((x - mean) / sd) ** 2)

    def prob(self, x, y):
        p = 1
        for a in range(len(x)):
            p *= self.gaussian(x[a], self.X_train[:, a], y)
        return p
        
    def predict(self, new_x):
        predicted = []
        for x in new_x:
            if self.prob(x, 1) > self.prob(x, -1):
                predicted.append(1)
            else:
                predicted.append(-1)
        return predicted

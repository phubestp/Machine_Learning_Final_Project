import numpy as np
import math

class SVM:

    def __init__(self, d, kernel='linear', C=1, iteration=1000, learning_rate=0.001) -> None:
        self.d = d
        self.C = C
        self.kernel = kernel
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.w = np.zeros(d)
        self.b = 0
    
    def hinge_loss(self):
        return np.dot(self.w, self.w) + self.C * sum([max(1 - y * (np.dot(self.w, x) + self.b), 0) for (x, y) in zip(self.X_train, self.y_train)]) 
    
    # def kernel_rbf(self, X, Z):
    #     self.gamma =  1 / (self.d * np.var(X)) 
    #     K = np.zeros((X.shape[0], Z.shape[0]))
    #     for i in range(len(X)):
    #         for j in range(len(Z)):
    #             K[i, j] = np.exp(-1*self.gamma*np.linalg.norm(X[i] - Z[j])**2)
    #     return K
    
    def calc_grad(self):
        dw = np.zeros(self.X_train.shape[1])
        db = 0
        for (x, y) in zip(self.X_train, self.y_train): 
            if y * (np.dot(self.w, x) + self.b) < 1:
                dw += self.w - (self.C * y * x)
                db += -1 * self.C * y
            else:
                dw += self.w
        return dw, db
    
    def grad_desc(self):
        # self.w = np.zeros(self.X_train.shape[1])
        # self.b = 0
        loss = math.inf
        for i in range(self.iteration):
            dw, db = self.calc_grad()
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if self.hinge_loss() < loss:
                loss = self.hinge_loss()
            else:
                break
        print(loss)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        # if self.kernel == 'rbf':
        #     self.X_train = self.kernel_rbf(X_train, X_train)

    def predict(self, X_test):
        self.grad_desc()
        X = X_test
        if self.kernel == 'rbf':
            X = self.kernel_rbf(X_test, X_test)
            print("YES")
        predicted = []
        for x in X:
            if -1*np.dot(self.w, x) + self.b > 0:
                predicted.append(-1)
            else:
                predicted.append(1)
        return predicted

    def score(self, pred, y_test):
        correct = 0
        for (p, y) in zip(pred, y_test):
            if p == y: correct += 1
        return correct/len(pred)
    

        
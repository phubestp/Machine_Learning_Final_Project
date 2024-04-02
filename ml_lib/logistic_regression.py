import math
import numpy as np

class LogisticRegression:

    def __init__(self, d, iteration=1000, learning_rate=0.001) -> None:
        self.d = d
        self.w = np.zeros(d)
        self.b = 0
        self.iteration = iteration
        self.learning_rate = learning_rate

    def calc_grad(self, y_train):
        dw = np.zeros(self.d)
        db = 0
        for (x, y) in zip(self.X_train, y_train):
            dw += x*y*(np.exp(y*np.dot(x.reshape(1, -1), self.w)[0] + self.b)) / ((np.exp(y*np.dot(x.reshape(1, -1), self.w)[0] + self.b) + 1))
            db += y*(np.exp(y*np.dot(x.reshape(1, -1), self.w)[0] + self.b)) / ((np.exp(y*np.dot(x.reshape(1, -1), self.w)[0] + self.b) + 1))
        return dw, db
    
    def grad_desc(self, y_train):
        self.w = np.zeros(self.d)
        self.b = 0
        for i in range(self.iteration):
            dw, db = self.calc_grad(y_train)
            self.w -= self.learning_rate*dw
            self.b -= self.learning_rate*db

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, y_train=None):
        if len(set(self.y_train)) == 2:
            self.grad_desc(self.y_train)
        else:
            self.grad_desc(y_train)
        predicted = []
        for x in X_test:
            pos = 1/(1+np.exp((1*((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            neg = 1/(1+np.exp((-1*((np.dot(x.reshape(1, -1), self.w) + self.b)))[0]))
            if len(set(self.y_train)) > 2:
                predicted.append(pos)
                continue
            if pos > neg:
                predicted.append(1)
            else: 
                predicted.append(-1)
        return predicted
    
    # def pred_multiclass(self, X_test):
    #     actual_pred = []
    #     all_pred = []

    #     for c in set(self.y_train):
    #         X = self.X_train
    #         y = np.where(self.y_train == c, 1, -1)
    #         self.grad_desc(y)
    #         all_pred.append(self.predict(X_test, y))
        
    #     all_pred = np.array(all_pred)

    #     for i in range(X_test.shape[0]):            
    #         p = all_pred[: ,i]
    #         actual_pred.append(np.argmax(p))

    #     return actual_pred

    def score(self, pred, y_test):
        correct = 0
        for (p, y) in zip(pred, y_test):
            if p == y: correct += 1
        return correct/len(pred)
             
import numpy as np

class TrainTestSplit:

    def __init__(self) -> None:
        pass

    def train_test_split(self, X, y, test_size=0.33, random_state=42):
        np.random.seed(random_state)
        test_index = np.random.choice(len(X), round(test_size * len(X)))
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(len(X)):
            if(i in test_index): 
                x_test.append(X[i])
                y_test.append(y[i])
            else: 
                x_train.append(X[i])
                y_train.append(y[i])
        return [np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)]
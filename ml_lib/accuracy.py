class Accuracy:

    def __init__(self) -> None:
        pass

    def score(self, pred, y_test):
        correct = 0
        for i in range(len(y_test)):
            if pred[i] == y_test[i]: correct += 1
        return correct / len(y_test)
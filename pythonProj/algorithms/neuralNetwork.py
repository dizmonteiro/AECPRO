import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


class NeuralNetwork:
    def __init__(self, train, test):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.nn = MLPClassifier(random_state=1, max_iter=300)

    def fit(self):
        self.nn.fit(self.x_train, self.y_train)

    def predict(self):
        predictions = self.nn.predict(self.x_test)
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))

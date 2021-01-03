import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

class _LogisticRegression:

    def __init__(self, train, test):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.log_model = LogisticRegression()

    def fit(self):
        self.log_model.fit(self.x_train, self.y_train)

    def predict(self):
        predictions = self.log_model.predict(self.x_test)

        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))
        print(accuracy_score(self.y_test, predictions))



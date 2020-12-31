import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class _LogisticRegression:

    def __init__(self):
        self.training = pd.read_csv('ProcessedDataTraining.csv')
        self.test = pd.read_csv('ProcessedDataTest.csv')

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.log_model = LogisticRegression()

    def fit(self):
        self.log_model.fit(self.x_train, self.y_train)

    def predict(self):
        predictions = self.log_model.predict(self.x_test)

        # Note: 79.9% accuracy   Todo: Rerun if data changed
        print(confusion_matrix(self.y_test, predictions))



import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree


class decisionTree:

    def __init__(self):
        self.training = pd.read_csv('ProcessedDataTraining.csv')
        self.test = pd.read_csv('ProcessedDataTest.csv')

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.dtc = tree.DecisionTreeClassifier()

    def fit(self):
        self.dtc = self.dtc.fit(self.x_train, self.y_train)

    def predict(self):
        predictions = self.dtc.predict(self.x_test)
        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


class SupportVectorMachines:
    def __init__(self, train, test):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.model = SVC()

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        predictions = self.model.predict(self.x_test)
        print(confusion_matrix(self.y_test, predictions))

        print(classification_report(self.y_test, predictions))

    # ToDo change func name
    def iterationsOptimize(self):
        param_grid = {'C': [1], 'gamma': [1], 'kernel': ['rbf']}
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        grid.fit(self.x_train, self.y_train)

        print(grid.best_params_)
        print(grid.best_estimator_)

        grid_predictions = grid.predict(self.x_test)

        print(confusion_matrix(self.y_test, grid_predictions))

        print(classification_report(self.y_test, grid_predictions))

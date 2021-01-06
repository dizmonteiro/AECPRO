import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class NeuralNetwork:
    def __init__(self, train, test, parameters):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(20,), learning_rate='adaptative',
                                solver='adam')

    def gridSearch(self):
        parameter_space = {
            'hidden_layer_sizes': [(10, 30, 10), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant'],
        }
        clf = GridSearchCV(self.nn, parameter_space, n_jobs=-1, cv=2)
        clf.fit(self.x_train, self.y_train)
        print('Best parameters found:\n', clf.best_params_)
        print('multimetric \n',clf.cv_results_)
        # {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (20,), 'learning_rate': 'adaptive',
        # 'solver': 'adam'}

    def fit(self):
        self.nn = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(20,), solver='adam', learning_rate='adaptive')
        self.nn.fit(self.x_train, self.y_train)

    def predict(self):
        pred = self.nn.predict(self.x_test)

        print(confusion_matrix(self.y_test, pred))
        print(classification_report(self.y_test, pred))

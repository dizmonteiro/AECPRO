import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

class EnsembleLearner:
    def __init__(self, train, test):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.kfold = model_selection.KFold(n_splits=2)
        self.estimators = []

    def fit(self):
        #model1 = KNeighborsClassifier()
        #self.estimators.append(('knn', model1))

        model2 = DecisionTreeClassifier()
        self.estimators.append(('cart', model2))

        model3 = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(20,), learning_rate='adaptive',
                                solver='adam')
        self.estimators.append(('nn', model3))

        #model4 = LogisticRegression()
        #self.estimators.append(('lr', model4))

        model5 = SVC()
        self.estimators.append(('svc', model5))

    def predict(self):
        model = VotingClassifier(self.estimators)
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)

        print(confusion_matrix(self.y_test, predictions))
        print(classification_report(self.y_test, predictions))







# https://www.datacamp.com/community/tutorials/ensemble-learning-python

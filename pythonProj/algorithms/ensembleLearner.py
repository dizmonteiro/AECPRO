import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
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

        self.kfold = model_selection.KFold(n_splits=2, random_state=42)
        self.estimators = []

    def fit(self):
        model1 = LogisticRegression()
        self.estimators.append(('logistic', model1))

        model2 = DecisionTreeClassifier()
        self.estimators.append(('cart', model2))

        model3 = SVC()
        self.estimators.append(('svm', model3))

    def predict(self):
        ensemble = VotingClassifier(self.estimators)
        results = model_selection.cross_val_score(ensemble, self.x_train, self.y_train, cv=self.    kfold)
        print(results.mean())






# https://www.datacamp.com/community/tutorials/ensemble-learning-python

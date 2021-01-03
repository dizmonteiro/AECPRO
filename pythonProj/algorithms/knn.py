import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


class KNN:
    def __init__(self, train, test):
        self.training = pd.read_csv(train)
        self.test = pd.read_csv(test)

        self.x_train = self.training.drop(' salary-classification', axis=1)
        self.y_train = self.training[' salary-classification']
        self.x_test = self.test.drop(' salary-classification', axis=1)
        self.y_test = self.test[' salary-classification']

        self.knn = KNeighborsClassifier()

    def fit(self, k):
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.x_train, self.y_train)

    # ToDo change name
    # define range dinamic
    def idealK(self):
        # Note: k ideal = 20
        error_rate = []
        for i in range(1, 100):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(self.x_train, self.y_train)
            pred_i = knn.predict(self.x_test)
            error_rate.append(np.mean(pred_i != self.y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 100), error_rate, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()

    def predict(self):
        pred = self.knn.predict(self.x_test)

        print(confusion_matrix(self.y_test, pred))
        print(classification_report(self.y_test, pred))

# scaler = StandardScaler()
# scaler.fit(training.drop(' salary-classification', axis=1))

# note: should binary data be scaled?
# scaled_features_training = scaler.transform(training.drop(' salary-classification',axis=1))
# scaled_features_testing = scaler.transform(test.drop(' salary-classification',axis=1))
# scaled_features_training = pd.DataFrame(scaled_features_training,columns=training.columns[:-1])
# scaled_features_testing = pd.DataFrame(scaled_features_testing,columns=test.columns[:-1])

# x_train = np.delete(scaled_features_training, 6, 1)
# y_train = scaled_features_training[:, 6]
# x_test = np.delete(scaled_features_testing, 6, 1)
# y_test = scaled_features_testing[:, 6]

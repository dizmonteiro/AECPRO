import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

training = pd.read_csv('ProcessedDataTraining.csv')
test = pd.read_csv('ProcessedDataTest.csv')

x_train = training.drop(' salary-classification', axis=1)
y_train = training[' salary-classification']
x_test = test.drop(' salary-classification', axis=1)
y_test = test[' salary-classification']

log_model = LogisticRegression()
log_model.fit(x_train, y_train)

predictions = log_model.predict(x_test)

# Note: 79.9% accuracy   Todo: Rerun if data changed
print(confusion_matrix(y_test, predictions))

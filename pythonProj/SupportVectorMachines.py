import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

training = pd.read_csv('ProcessedDataTraining.csv')
test = pd.read_csv('ProcessedDataTest.csv')

x_train = training.drop(' salary-classification', axis=1)
y_train = training[' salary-classification']
x_test = test.drop(' salary-classification', axis=1)
y_test = test[' salary-classification']

model = SVC()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test,predictions))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# -- Source do dataset original
# https://www.kaggle.com/uciml/adult-census-income

# -- Links possivelmente uteis
# -> https://datatofish.com/select-rows-pandas-dataframe/

# -- var do tipo pandas.core.frame.DataFrame
training = pd.read_csv('training.csv', sep=';', engine='python')

# -- alguma informação sobre o dataset
# print(training.shape)
# print(training.info())
# print(training.describe())

# -- Pairplot - pesado - evitar
# sns.pairplot(training)
# plt.show()

# -- Histograma relativo a capital-gain
# -- Capital Gain - diferença entre o valor de revenda de um bem e o seu valor de compra
# training[' capital-gain'].plot(kind='hist')
# plt.show()

# -- first model
training = training.replace(' <=50K', 0)
training = training.replace(' >=50K', 10)
training = training.replace(' >50K', 10)
X = training[['age', ' fnlwgt', ' education-num', ' capital-gain', ' capital-loss', ' hours-per-week']]
y = training[' salary-classification']


lm = LinearRegression()
lm.fit(X, y)

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# -- predicts
test = pd.read_csv('test.csv', sep=';', engine='python')
test = test.replace(' <=50K', 0)
test = test.replace(' >=50K', 10)
test = test.replace(' >50K', 10)

X_test = training[['age', ' fnlwgt', ' education-num', ' capital-gain', ' capital-loss', ' hours-per-week']]
y_test = training[' salary-classification']
predictions = lm.predict(X_test)
# plt.scatter(y_test, predictions)
# sns.distplot((y_test-predictions), bins=50);
# plt.show()
print(metrics.mean_absolute_error(y_test, predictions))
np.savetxt('LinearRegressionPrediction.csv', predictions, delimiter=',')

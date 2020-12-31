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
class _LinearRegression:
        
    def __init__(self):
        # -- var do tipo pandas.core.frame.DataFrame
        self.training = pd.read_csv('processedDataTraining.csv', sep=',', engine='python')
        self.lm = LinearRegression()
    
    def datasetInfo(self):
        # -- Alguma informação sobre o dataset
        print(self.training.shape)
        print(self.training.info())
        print(self.training.describe())

    def pairplot(self):
        # -- Pairplot - pesado - evitar
        sns.pairplot(self.training)
        plt.show()    

    # -- Histograma relativo a capital-gain
    # -- Capital Gain - diferença entre o valor de revenda de um bem e o seu valor de compra
    # training[' capital-gain'].plot(kind='hist')
    # plt.show()

    # -- note: check if its needed
    #training = training.replace(' <=50K', 0)
    #training = training.replace(' >=50K', 10)
    #training = training.replace(' >50K', 10)
    def fit(self):
        X = self.training.drop([' salary-classification'], axis=1
                    )
        y = self.training[' salary-classification']
        
        self.lm.fit(X,y)  

        print(self.lm.intercept_)
        coeff_df = pd.DataFrame(self.lm.coef_, X.columns, columns=['Coefficient'])
        print(coeff_df)

    def predict(self):
        # -- predicts
        test = pd.read_csv('processedDataTest.csv', sep=',', engine='python')
        test = test.replace(' <=50K', 0)
        test = test.replace(' >=50K', 10)
        test = test.replace(' >50K', 10)

        X_test = test.drop([' salary-classification'], axis=1)
        y_test = test[' salary-classification']
        predictions = self.lm.predict(X_test)
        # plt.scatter(y_test, predictions)
        # sns.distplot((y_test-predictions), bins=50);
        # plt.show()
        print(metrics.mean_absolute_error(y_test, predictions))

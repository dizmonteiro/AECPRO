import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- Source do dataset original
# https://www.kaggle.com/uciml/adult-census-income

# -- var do tipo pandas.core.frame.DataFrame
training = pd.read_csv('training.csv', sep=';', engine='python')

# -- alguma informação sobre o dataset
#print(training.shape)
#print(training.info())
#print(training.describe())

# -- Pairplot - pesado - evitar
#sns.pairplot(training)
#plt.show()

# -- Histograma relativo a capital-gain
# -- Capital Gain - diferença entre o valor de revenda de um bem e o seu valor de compra
#training[' capital-gain'].plot(kind='hist')
#plt.show()
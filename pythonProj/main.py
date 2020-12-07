import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -- Source do dataset original
# https://www.kaggle.com/uciml/adult-census-income

# -- var do tipo pandas.core.frame.DataFrame
training = pd.read_csv('training.csv', sep=';', engine='python')

#print(training.shape)
print(training.info())
print(training.describe())

# -- Pairplot - pesado
#sns.pairplot(training)
#plt.show()
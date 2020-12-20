import pandas as pd
import matplotlib.pyplot as plt
# -- ToDo: mudar de volta para "training.csv"
dataset = pd.read_csv('training.csv', sep=';', engine='python')

# -- ToDo: reduzir numero de workclasses e occupation
# -- ToDo: ver numero de diferentes native country's. Se necessário reduzir quantidade

# -- Note: 15 different occupations
# occupationCount = dataset[' occupation'].value_counts()
# occupationCount.plot.bar()
# plt.show()

# -- Note: 6 different relationship status
# relationshipCount = dataset[' relationship'].value_counts()
# relationshipCount.plot.bar()
# plt.show()

# -- Note: 5 different races
# raceCount = dataset[' race'].value_counts()
# raceCount.plot.bar()
# plt.show()

# -- Note: 5 different races
# raceCount = dataset[' race'].value_counts()
# raceCount.plot.bar()
# plt.show()

# -- Note: 42 different native country's
# -- Awful distribution - ignore maybe?
# countryCount = dataset[' native-country'].value_counts()
# countryCount.plot.bar()
# plt.show()

# -- Note: 2 different sex's
# -- Dataset misaligned - approximately 2x more males
# sexCount = dataset[' sex'].value_counts()
# sexCount.plot.bar()
# plt.show()

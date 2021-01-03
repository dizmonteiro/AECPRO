import pandas as pd
import matplotlib.pyplot as plt

# -- ToDo: mudar de volta para "training.csv"
training = pd.read_csv('data/training.csv', sep=';', engine='python', header='infer')
test = pd.read_csv('data/test.csv', sep=';', engine='python', header='infer')

training = training.replace(' <=50K', 0)
training = training.replace(' >=50K', 1)
training = training.replace(' >50K', 1)
test = test.replace(' <=50K', 0)
test = test.replace(' >=50K', 1)
test = test.replace(' >50K', 1)

# -- ToDo: reduzir numero de workclasses e occupation
# -- ToDo: ver numero de diferentes native country's. Se necessário reduzir quantidade

# -- Note: 15 different occupations
occupationCount = training[' occupation'].value_counts()
occupationCount.plot.bar()
plt.show()

# -- Note: 6 different relationship status
relationshipCount = training[' relationship'].value_counts()
relationshipCount.plot.bar()
plt.show()

# -- Note: 5 different races
raceCount = training[' race'].value_counts()
raceCount.plot.bar()
plt.show()

# -- Note: 42 different native country's
# -- Awful distribution - ignore maybe?
countryCount = training[' native-country'].value_counts()
countryCount.plot.bar()
plt.show()

# -- Note: 2 different sex's
# -- training misaligned - approximately 2x more males
sexCount = training[' sex'].value_counts()
sexCount.plot.bar()
plt.show()

# ----------------------- PROCESSING -----------------------
# -- Note: Reduce countrys present due to poor distribution
for index, row in training.iterrows():
    if row[' native-country'] != ' United-States':
        training.loc[index, ' native-country'] = ' Other'
for index, row in test.iterrows():
    if row[' native-country'] != ' United-States':
        test.loc[index, ' native-country'] = ' Other'
# ----------------------- ANALYSIS -----------------------
# -- NATIVE COUNTRY --
print("\nNative Country:")
nCountry = training.loc[training[' native-country'] == ' United-States']
print("\t United-States: %f" % nCountry[' salary-classification'].mean())
nCountry = training.loc[training[' native-country'] == ' Other']
print("\t Other: %f" % nCountry[' salary-classification'].mean())

# -- SEX --
print("\nSex:")
nCountry = training.loc[training[' sex'] == ' Male']
print("\t Male: %f" % nCountry[' salary-classification'].mean())
nCountry = training.loc[training[' sex'] == ' Female']
print("\t Female: %f" % nCountry[' salary-classification'].mean())

# --------------- Drop Columns ---------------
training = training.drop([' education'], axis=1)
test = test.drop([' education'], axis=1)

# -- DUMMIES --
training = pd.get_dummies(training)
test = pd.get_dummies(test)

print(training.head().to_string())



# --------------------- Export ---------------------
training.to_csv('data/processedDataTraining.csv', index=False)
test.to_csv('data/processedDataTest.csv', index=False)

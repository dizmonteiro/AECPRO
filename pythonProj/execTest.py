from algorithms.SupportVectorMachines import SupportVectorMachines
from algorithms.ensembleLearner import EnsembleLearner
from algorithms.neuralNetwork import NeuralNetwork

lr = EnsembleLearner('data/processedDataTraining_under.csv', 'data/processedDataTest.csv')
#lr.gridSearch()
lr.fit()
lr.predict()
#lr.PCA()

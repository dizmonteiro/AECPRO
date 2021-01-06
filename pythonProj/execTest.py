from algorithms.SupportVectorMachines import SupportVectorMachines
from algorithms.ensembleLearner import EnsembleLearner
from algorithms.neuralNetwork import NeuralNetwork

lr = NeuralNetwork('data/processedDataTraining.csv', 'data/processedDataTest.csv', '')
lr.gridSearch()
#lr.fit()
#lr.predict()

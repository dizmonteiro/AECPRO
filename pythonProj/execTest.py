from algorithms.SupportVectorMachines import SupportVectorMachines

lr = SupportVectorMachines('data/processedDataTraining2.csv', 'data/processedDataTest.csv')
lr.fit()
lr.predict()

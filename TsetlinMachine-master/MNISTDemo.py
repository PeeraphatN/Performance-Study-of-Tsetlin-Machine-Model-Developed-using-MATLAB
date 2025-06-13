import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import MultiClassTsetlinMachine

# Parameters of the pattern recognition problem
number_of_features = 784
number_of_classes = 10

# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = number_of_classes * 5
states = 100 

# Training configuration
epochs = 200

# Loading of training and test data
training_data = np.loadtxt("C:\Work\Research\Project\DataSet\MNIST\MNISTTraining.txt").astype(dtype=np.int32)
test_data = np.loadtxt("C:\Work\Research\Project\DataSet\MNIST\MNISTTest.txt").astype(dtype=np.int32)
X_training = training_data[:,0:784] # Input features
y_training = training_data[:,784] # Class labels

X_test = test_data[:,0:784] # Input features
y_test = test_data[:,784] # Class labels

for n in range(10):
    for i in range(28):
        print("28 * 28 mnist: ", X_training[n][i*28:(i+1)*28])
    print("Class label: ", y_training[n])
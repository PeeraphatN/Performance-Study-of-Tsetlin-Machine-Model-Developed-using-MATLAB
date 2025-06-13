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
number_of_clauses = 100
states = 100

# Training configuration
epochs = 400

# Loading of training and test data
training_data = np.loadtxt(r"C:\Work\Research\Project\DataSet\MNIST\MNISTTraining.txt").astype(dtype=np.int32)
test_data = np.loadtxt(r"C:\Work\Research\Project\DataSet\MNIST\MNISTTest.txt").astype(dtype=np.int32)
X_training = training_data[:,0:784] # Input features
y_training = training_data[:,784] # Class labels

X_test = test_data[:,0:784] # Input features
y_test = test_data[:,784] # Class labels

tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

starttime = np.datetime64("now")
print ("Training the Tsetlin Machine on MNIST data ...")

tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

print ("Training completed. total time used:", np.datetime64("now") - starttime)
print ("Hyperparameters:")
print ("Number of features:", number_of_features)
print ("Number of classes:", number_of_classes)
print ("T:", T)
print ("s:", s)
print ("Number of clauses:", number_of_clauses)
print ("Number of states:", states)
print ("epochs:", epochs)
print ("Number of training samples:", y_training.shape[0])
print ("Number of test samples:", y_test.shape[0])
print ("\nEvaluating the Tsetlin Machine on test and training data...\n")
print ("Accuracy on test data:", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
print ("Accuracy on training data:", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
print ("\nPredictions for first 20 test samples:")
for i in range(20):
    print(f"Sample {i+1}: Predicted class = {tsetlin_machine.predict(X_test[i])}, True class = {y_test[i]}")
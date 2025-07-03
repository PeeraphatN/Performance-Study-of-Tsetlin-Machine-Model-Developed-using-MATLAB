import numpy as np
import argparse
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import MultiClassTsetlinMachine

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=15, help="Threshold (T)")
parser.add_argument("--s", type=float, default=3.9, help="Sensitivity (s)")
parser.add_argument("--clauses", type=int, help="Number of Clauses")
parser.add_argument("--states", type=int, default=100, help="Number of States")
parser.add_argument("--epochs", type=int, default=200, help="Number of Epochs")
args = parser.parse_args()

# Parameters of the pattern recognition problem
number_of_features = 784
number_of_classes = 10

# Parameters for the Tsetlin Machine
T = args.T
s = args.s
number_of_clauses = args.clauses
states = args.states

# Training configuration
epochs = args.epochs

# Loading of training and test data
training_data = np.loadtxt(r"C:\Work\Research\Project\DataSet\XOR\Noisy\NoisyXORTrainingData.csv",delimiter=',').astype(dtype=np.int32)
test_data = np.loadtxt(r"C:\Work\Research\Project\DataSet\XOR\Noisy\NoisyXORTestData.csv", delimiter=',').astype(dtype=np.int32)

X_training = training_data[:,0:number_of_features] # Input features
y_training = training_data[:,number_of_features] # Target value

X_test = test_data[:,0:number_of_features] # Input features
y_test = test_data[:,number_of_features] # Target value

# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
print ("Training the Tsetlin Machine on MNIST data ...")
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

starttime = np.datetime64("now")
tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
duration = (np.datetime64("now") - starttime) / np.timedelta64(1, 's')
print ("Training completed. total time used:", (np.datetime64("now") - starttime) / np.timedelta64(1, 's'))

print("\nEvaluating the Tsetlin Machine on test and training data...\n\n")

# Some performance statistics

print ("Accuracy on test data (no noise):", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]))
print ("Accuracy on training data (40% noise):", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0]))
print ("Prediction: x1 = 1, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([1,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([0,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 0, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([0,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))
print ("Prediction: x1 = 1, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([1,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32)))

import os
import csv

os.makedirs("result\\noisy_xor", exist_ok=True)

csv_path = os.path.join("result","noisy_xor","noisy_xor_result_log.csv")

test_acc = tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
train_acc = tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])

row = {
    "number_of_features": number_of_features,
    "number_of_classes": number_of_classes,
    "T": T,
    "s": s,
    "number_of_clauses": number_of_clauses,
    "number_of_states": states,
    "epochs": epochs,
    "Accuracy on test data": round(test_acc, 4),
    "Accuracy on training data": round(train_acc, 4),
    "Time": duration
}

file_exists = os.path.isfile(csv_path)
with open(csv_path, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Result logged to: {csv_path}")
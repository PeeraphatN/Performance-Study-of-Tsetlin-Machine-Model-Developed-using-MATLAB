import argparse
from pathlib import Path

import numpy as np

import pyximport

BASE_DIR = Path(__file__).resolve().parent
PYX_BUILD_DIR = BASE_DIR / "pyxbld"
PYX_BUILD_DIR.mkdir(parents=True, exist_ok=True)

pyximport.install(
    setup_args={"include_dirs": np.get_include()},
    build_dir=str(PYX_BUILD_DIR),
    reload_support=True,
)

import MultiClassTsetlinMachine

# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=15, help="Threshold (T)")
parser.add_argument("--s", type=float, default=3.9, help="Sensitivity (s)")
parser.add_argument("--clauses", type=int,default=10 , help="Number of Clauses")
parser.add_argument("--states", type=int, default=100, help="Number of States")
parser.add_argument("--epochs", type=int, default=200, help="Number of Epochs")
args = parser.parse_args()

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Parameters for the Tsetlin Machine
T = args.T
s = args.s
number_of_clauses = args.clauses
states = args.states

# Training configuration
epochs = args.epochs

# Loading of training and test data
data_dir = BASE_DIR.parent / "DataSet" / "XOR" / "Noisy"
training_data_path = data_dir / "NoisyXORTrainingData.csv"
test_data_path = data_dir / "NoisyXORTestData.csv"

training_data = np.loadtxt(training_data_path, delimiter=',').astype(dtype=np.int32)
test_data = np.loadtxt(test_data_path, delimiter=',').astype(dtype=np.int32)

X_training = training_data[:,0:number_of_features] # Input features
y_training = training_data[:,number_of_features] # Target value

X_test = test_data[:,0:number_of_features] # Input features
y_test = test_data[:,number_of_features] # Target value

# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
print ("Training the Tsetlin Machine on NoisyXOR data ...")
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
acc_log = tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)
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

import csv

result_dir = BASE_DIR / "result" / "noisy_xor"
result_dir.mkdir(parents=True, exist_ok=True)

csv_path = result_dir / "noisy_xor_result_log.csv"

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

file_exists = csv_path.exists()
with csv_path.open(mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=row.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(row)

print(f"Result logged to: {csv_path}")

import pandas as pd
from datetime import datetime

# Create log directory if not exists
log_dir = result_dir

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
epoch_log_path = log_dir / f"epoch_log_{timestamp}.csv"

# Save epoch accuracy log to CSV
epoch_df = pd.DataFrame({
    "epoch": np.arange(1, len(acc_log) + 1),
    "accuracy": acc_log
})
epoch_df.to_csv(epoch_log_path, index=False)
print(f"Epoch accuracy log saved to: {epoch_log_path}")

import pandas as pd
import sys
import os

def detect_dataset_type(filepath):
    with open(filepath, 'r') as file:
        first_line = file.readline()
        parts = first_line.strip().split()
        num_columns = len(parts)
    
    if num_columns == 13:
        return 'XOR'
    elif num_columns == 785:
        return 'MNIST'
    else:
        raise ValueError(f"Unrecognized file format with {num_columns} columns")

def convert_xor(filepath, output_path):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = list(map(int, line.strip().split()))
            features = parts[:-1]
            label = parts[-1]
            data.append(features + [label])
    
    columns = [f'feature_{i+1}' for i in range(12)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"✔ XOR dataset converted to: {output_path}")

def convert_mnist(filepath, output_path):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = list(map(int, line.strip().split()))
            features = parts[:-1]
            label = parts[-1]
            data.append(features + [label])
    
    columns = [f'pixel_{i+1}' for i in range(784)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"✔ MNIST dataset converted to: {output_path}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_dataset.py <input_txt_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    dataset_type = detect_dataset_type(input_file)
    output_file = os.path.splitext(input_file)[0] + ".csv"

    if dataset_type == 'XOR':
        convert_xor(input_file, output_file)
    elif dataset_type == 'MNIST':
        convert_mnist(input_file, output_file)

if __name__ == '__main__':
    main()

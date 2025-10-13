# How to Use: Tsetlin Machine Demos (MATLAB + Python)

This repo provides ready-to-run demos of the Tsetlin Machine in MATLAB and Python. Use it to train on MNIST and XOR variants, log results, and generate comparison plots.

## Prerequisites
- MATLAB R2019b+ with `-batch` support (for CLI runs)
- Python 3.8+ with `pip`
- Recommended packages: `numpy`, `pandas`, `matplotlib`

Install Python deps (if needed):
```
pip install -r requirements.txt  # if present
pip install numpy pandas matplotlib
```

## Repo Layout
- `DataSet/` sample datasets (MNIST, XOR variants)
- `MATLAB/` MATLAB demos: `MNIST.m`, `NoisyXOR.m`, `NormalXOR.m` (+ batch runners)
- `TsetlinMachine-master/` Python/C demos: `MNISTDemo.py`, `NoisyXORDemo.py`
- `TsetlinMachine-purePython/` Pure-Python demos: `MNISTPure.py`, `NoisyXORPure.py`, `NormalXORPure.py`
- `result/` experiment outputs, logs, and plots

## Quick Start (most users)
1) Run a small XOR demo in Python (fast sanity check):
```
cd TsetlinMachine-purePython
python NoisyXORPure.py --clauses 10
```
2) Run MNIST in Python (C-extension demo):
```
cd TsetlinMachine-master
python MNISTDemo.py --clauses 100
```
3) Plot results (from repo root):
```
python plot_results_all.py
```
Plots appear under `result/<dataset>/plots/`.

## Run MATLAB Demos
From `MATLAB/` you can run each script directly:
```
cd MATLAB
matlab -batch "MNIST('clauses',100)"
matlab -batch "NoisyXOR('clauses',10)"
matlab -batch "NormalXOR('clauses',10)"
```
Batch over a clause range (Windows `.bat` helpers):
```
./run_mnist_dynamic.bat START STEP END
./run_noisy_xor_dynamic.bat START STEP END
./run_normal_xor_dynamic.bat START STEP END
```
Notes: `STEP` must not be `0`. If omitted, it defaults to `1`.

## Run Python Demos
Python with C-extension (faster):
```
cd TsetlinMachine-master
python MNISTDemo.py --clauses 100
python NoisyXORDemo.py --clauses 10
```
Pure-Python (reference):
```
cd TsetlinMachine-purePython
python MNISTPure.py --clauses 100
python NoisyXORPure.py --clauses 10
python NormalXORPure.py --clauses 10
```
Windows helpers to sweep clause ranges:
```
./run_mnist_dynamic.bat START STEP END       # in TsetlinMachine-master
./run_noisy_dynamic.bat START STEP END       # in TsetlinMachine-master
./run_mnist_pure.bat START STEP END          # launches MNISTDemo.py from master
./run_noisy_pure.bat START STEP END          # launches NoisyXORDemo.py from master
```

## Results and Logs
- Per-epoch logs: `*/result/<task>/epoch_log_YYYYMMDD_HHMMSS.csv`
- Summary logs: `*/result/*_result_log.csv`
- Most `result/` content is git-ignored. If logs are tracked already, remove from index to let `.gitignore` work:
```
git rm --cached path/to/result/file.csv
git commit -m "Stop tracking result logs"
```

## Plotting
From the repo root:
```
python plot_results_all.py
```
Options:
- `--metric clauses_per_second` (default) or `seconds_per_clause`
- `--agg mean|median`
- `--marker-step N` (use `0` for every point)

Outputs go to `result/<dataset>/plots/`.

## Adjustable Parameters
All demos expose the same core hyperparameters. Use them to balance accuracy, speed, and memory.

- `--T` or `T` (Threshold)
  - Default: 15
  - Meaning: Vote clipping threshold per class. Higher allows more clause votes before saturation; can improve capacity but may slow convergence.

- `--s` or `s` (Specificity/Sensitivity)
  - Default: 3.9
  - Meaning: Controls feedback probabilities. Lower tends to include more literals (more specific clauses); higher yields sparser clauses.

- `--clauses` or `clauses` (Number of Clauses)
  - Defaults: MNIST 100–500 depending on script; XOR 10
  - Meaning: Model capacity. More clauses can improve accuracy but increase compute and memory.

- `--states` or `states` (TA States per Literal)
  - Default: 100
  - Meaning: Automaton confidence granularity. More states smooth updates; uses more memory.

- `--epochs` or `epochs` (Training Epochs)
  - Defaults: MNIST 200; XOR 200–500 (script-dependent)
  - Meaning: Full passes over training set. Increase until accuracy plateaus.

How to pass parameters:
- Python: `python MNISTDemo.py --clauses 400 --T 25 --s 3.5 --states 200 --epochs 300`
- MATLAB: `matlab -batch "MNIST('clauses',400,'T',25,'s',3.5,'states',200,'epochs',300)"`

Notes:
- Script defaults vary slightly (check `--help` in Python demos).
- Very large `clauses × states` settings can consume significant RAM; scale gradually.

## Tips & Troubleshooting
- Use smaller `--clauses` to quickly validate end-to-end runs.
- On Windows, if `python` launches the Microsoft Store, try `py -3`.
- If MATLAB is not on PATH, use the full path to `matlab` executable.
- High parallel batch counts can be CPU/memory heavy; monitor usage.

## Need More?
Open an issue or ask for:
- Additional plots or custom aggregations in `plot_results_all.py`
- Standardized time logging across all scripts

# Performance Study of Tsetlin Machine Model (MATLAB + Python)

This repository contains MATLAB and Python implementations / demos of the Tsetlin Machine, example datasets, and conveniences (batch scripts) to run experiments across ranges of hyperparameters.

## Layout (important files/folders)
- `DataSet/` — datasets used by the demos (MNIST, XOR variants).
- `MATLAB/` — MATLAB implementations and scripts (`MNIST.m`, `NoisyXOR.m`, `NormalXOR.m`) and batch runners.
- `TsetlinMachine-master/` — Python/C demo using the compiled/extension implementation (demos: `MNISTDemo.py`, `NoisyXORDemo.py`).
- `TsetlinMachine-purePython/` — pure-Python demos (`MNISTPure.py`, `NoisyXORPure.py`, `NormalXORPure.py`) and helper batch files.
- `result/` and `*/result/` — experiment outputs and CSV logs (many are intentionally ignored by `.gitignore`).

## Quick goals
- Train and evaluate the Tsetlin Machine on datasets (MNIST, XOR variants).
- Produce per-epoch logs (CSV) and summary CSVs in `result/` folders.
- Provide batch scripts to run experiments across hyperparameter ranges.

## Running demos

Notes: examples below use PowerShell on Windows. Adjust `python` to a full path if you use a virtualenv.

### Python (TsetlinMachine-master)
Run a single demo from the `TsetlinMachine-master` folder:

```powershell
cd "c:\Tsetlin Machine\Performance-Study-of-Tsetlin-Machine-Model-Developed-using-MATLAB\TsetlinMachine-master"
python MNISTDemo.py --clauses 100
python NoisyXORDemo.py --clauses 10
```

Dynamic batch runner (launch many runs over a clause range):

```powershell
.\run_mnist_dynamic.bat START STEP END
.\run_noisy_dynamic.bat START STEP END
```

Examples:
```powershell
.\run_mnist_dynamic.bat 100 100 1000
.\run_noisy_dynamic.bat 2 2 20
```

### MATLAB
From the `MATLAB` folder you can run the MATLAB scripts directly (or use the batch runners that launch MATLAB instances):

```powershell
cd "c:\Tsetlin Machine\Performance-Study-of-Tsetlin-Machine-Model-Developed-using-MATLAB\MATLAB"
matlab -batch "MNIST('clauses',100)"
matlab -batch "NoisyXOR('clauses',10)"
matlab -batch "NormalXOR('clauses',10)"
```

Dynamic runner (launch multiple MATLAB runs):

```powershell
.\run_mnist_dynamic.bat START STEP END
.\run_noisy_xor_dynamic.bat START STEP END
.\run_normal_xor_dynamic.bat START STEP END
```

Important: the batch scripts validate `STEP` to avoid infinite loops; `STEP` must not be `0`. If `STEP` is omitted it defaults to `1`.

### Pure-Python demos
From the `TsetlinMachine-purePython` folder you can run the pure Python demos. Some batch files there launch the corresponding demos from `TsetlinMachine-master` instead (so the C-extension is used):

```powershell
cd "...\TsetlinMachine-purePython"
python MNISTPure.py --clauses 100
python NoisyXORPure.py --clauses 10
.\run_mnist_pure.bat START STEP END   # launches MNISTDemo.py in the master folder
.\run_noisy_pure.bat START STEP END   # launches NoisyXORDemo.py in the master folder
```

## Logs and results
- Per-run epoch logs are saved in `*/result/<task>/epoch_log_YYYYMMDD_HHMMSS.csv`.
- Summary run logs are appended to `*_result_log.csv` files inside the `result/` directories.
- The repository contains a `.gitignore` that excludes most `result/` folders and CSV logs. If you already have logs tracked in git, remove them from the index to let `.gitignore` take effect:

```powershell
git rm --cached path\to\result\file.csv
git commit -m "Remove tracked result logs"
```

## Adding training time to logs
Some scripts now include a `Time` column in their summary CSVs. If a script doesn't include it yet, you can add a duration field in the script right after training finishes. Example (Python):

```python
start = np.datetime64('now')
...
duration = (np.datetime64('now') - start) / np.timedelta64(1, 's')
row['Time'] = round(float(duration), 4)
```

## Safety & notes
- Batch scripts that launch many processes will consume system resources — run with care and monitor CPU/memory.
- If you want PowerShell-native runner scripts (jobs/background) instead of `start cmd /k`, say so and I can add them.

## Where to find help
- If you want me to add a plotting helper (MATLAB script) to summarize result logs into graphs, tell me which folder you want the script saved to (for example `MATLAB/result/mnist`) and I'll add it.
- If you want me to standardize `Time` logging across all scripts, I can make that change in a single commit.

---
Generated on: 2025-10-06
"# Performance-Study-of-Tsetlin-Machine-Model-Developed-using-MATLAB" 

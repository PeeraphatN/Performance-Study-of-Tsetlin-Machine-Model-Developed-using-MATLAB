#!/usr/bin/env python3
"""
plot_results_all.py

Scans the repository for result summary CSVs and epoch logs produced by MATLAB and Python demos
and generates comparison plots for: accuracy, training time, and per-epoch accuracy.

Produces PNG files under result/plots/ (created if missing).

Supported datasets: mnist, noisy_xor, normal_xor
Supported sources (if available): MATLAB (MATLAB/result/*), Python master (TsetlinMachine-master/result/*),
and pure-Python (TsetlinMachine-purePython/result/*).

The script is resilient to missing files and will skip absent entries.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from matplotlib.ticker import MaxNLocator


ROOT = os.path.abspath(os.path.dirname(__file__))
PLOTS_DIR = os.path.join(ROOT, "result", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# Fixed colors by method/source
# Orange for MATLAB, Blue for Python variants
COLOR_MAP = {
    'MATLAB': '#ff7f0e',         # orange
    'Python': '#1f77b4',         # blue
    'Python-master': '#1f77b4',  # blue
    'Python-pure': '#1f77b4',    # blue
}


def color_for(method: str):
    if method in COLOR_MAP:
        return COLOR_MAP[method]
    # Fallbacks for prefixes
    if method.startswith('Python'):
        return COLOR_MAP['Python']
    return None

def find_files(patterns):
    """Return a list of files matching any of the given glob patterns (relative to ROOT)."""
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(ROOT, p)))
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def read_summary(path):
    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            return None
        return df
    except Exception:
        return None


def read_epoch_log(path):
    try:
        # MATLAB writes header comment lines starting with '%'. Tell pandas to treat '%' as comment
        # so the actual CSV table is parsed correctly. Fall back to default read if needed.
        try:
            df = pd.read_csv(path, comment='%')
        except Exception:
            df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if 'epoch' in cols and 'accuracy' in cols:
            out = df[[cols['epoch'], cols['accuracy']]].rename(columns={cols['epoch']: 'epoch', cols['accuracy']: 'accuracy'})
            out['epoch'] = out['epoch'].astype(int)
            out['accuracy'] = pd.to_numeric(out['accuracy'], errors='coerce')
            return out
        if df.shape[1] >= 2:
            out = df.iloc[:, :2].rename(columns={df.columns[0]: 'epoch', df.columns[1]: 'accuracy'})
            out['epoch'] = out['epoch'].astype(int)
            out['accuracy'] = pd.to_numeric(out['accuracy'], errors='coerce')
            return out
        return None
    except Exception:
        return None


def read_all_epoch_logs(patterns):
    """Return list of epoch dfs for files matching patterns (newest first)."""
    files = find_files(patterns)
    dfs = []
    for f in files:
        df = read_epoch_log(f)
        if df is not None:
            dfs.append(df.set_index('epoch'))
    return dfs


def find_summary_and_epoch(dataset):
    """Return dict of method -> (summary_row, epoch_df or None)
    methods: MATLAB, Python-master, Python-pure
    """
    out = {}

    # MATLAB locations
    # MATLAB may write to either MATLAB/result/... or MATLAB/MATLAB/result/... depending on how scripts were run.
    matlab_summary_candidates = {
        'mnist': ['MATLAB/result/mnist/mnist_result_log.csv', 'MATLAB/MATLAB/result/mnist/mnist_result_log.csv'],
        'noisy_xor': ['MATLAB/result/noisy_xor/noisyXOR_result_log.csv', 'MATLAB/MATLAB/result/noisy_xor/noisyXOR_result_log.csv'],
        'normal_xor': ['MATLAB/result/normal_xor/normalXOR_result_log.csv', 'MATLAB/MATLAB/result/normal_xor/normalXOR_result_log.csv'],
    }[dataset]

    matlab_epoch_patterns_candidates = {
        'mnist': [['MATLAB/result/mnist/*epoch_log*.csv', 'MATLAB/result/mnist/mnist_epoch_log_*.csv'], ['MATLAB/MATLAB/result/mnist/*epoch_log*.csv', 'MATLAB/MATLAB/result/mnist/mnist_epoch_log_*.csv']],
        'noisy_xor': [['MATLAB/result/noisy_xor/*epoch_log*.csv', 'MATLAB/result/noisy_xor/noisy_xor_epoch_log_*.csv'], ['MATLAB/MATLAB/result/noisy_xor/*epoch_log*.csv', 'MATLAB/MATLAB/result/noisy_xor/noisy_xor_epoch_log_*.csv']],
        'normal_xor': [['MATLAB/result/normal_xor/*epoch_log*.csv', 'MATLAB/result/normal_xor/normal_xor_epoch_log_*.csv'], ['MATLAB/MATLAB/result/normal_xor/*epoch_log*.csv', 'MATLAB/MATLAB/result/normal_xor/normal_xor_epoch_log_*.csv']],
    }[dataset]

    # pick the first existing summary path (if any)
    matlab_summary = None
    for p in matlab_summary_candidates:
        if os.path.isfile(os.path.join(ROOT, p)):
            matlab_summary = p
            break

    # choose epoch patterns by checking which candidate directory contains matches; merge both candidates so we find all files
    matlab_epoch_patterns = []
    for group in matlab_epoch_patterns_candidates:
        for pat in group:
            # keep pattern even if no files exist so read_all_epoch_logs will check them
            matlab_epoch_patterns.append(pat)

    # Python master locations -- removed per user request

    # Pure Python locations
    pure_summary = {
        'mnist': 'TsetlinMachine-purePython/result/mnist_pure_python/mnist_pure_python_result_log.csv',
        'noisy_xor': 'TsetlinMachine-purePython/result/noisy_xor_pure_python/noisy_xor_pure_python_result_log.csv',
        'normal_xor': 'TsetlinMachine-purePython/result/normal_xor_pure_python/normal_xor_pure_python_result_log.csv',
    }.get(dataset)
    pure_epoch_patterns = {
        'mnist': ['TsetlinMachine-purePython/result/mnist_pure_python/*epoch_log*.csv'],
        'noisy_xor': ['TsetlinMachine-purePython/result/noisy_xor_pure_python/*epoch_log*.csv'],
        'normal_xor': ['TsetlinMachine-purePython/result/normal_xor_pure_python/*epoch_log*.csv'],
    }[dataset]

    # MATLAB
    if matlab_summary and os.path.isfile(os.path.join(ROOT, matlab_summary)):
        s = read_summary(os.path.join(ROOT, matlab_summary))
    else:
        s = None
    e_dfs = read_all_epoch_logs(matlab_epoch_patterns)
    out['MATLAB'] = (s, e_dfs)

    # Python master: omitted

    # Python pure
    if pure_summary and os.path.isfile(os.path.join(ROOT, pure_summary)):
        s = read_summary(os.path.join(ROOT, pure_summary))
    else:
        s = None
    e_dfs = read_all_epoch_logs(pure_epoch_patterns)
    out['Python-pure'] = (s, e_dfs)

    return out


def aggregate_epoch_dfs(dfs, agg='mean'):
    """Given a list of epoch-indexed dfs, return aggregated series (index=epoch) using agg (mean/median) and std."""
    if not dfs:
        return None, None
    # concat along columns, aligning on epoch index
    concat = pd.concat([df['accuracy'].rename(f'run{i}') for i, df in enumerate(dfs)], axis=1)
    if agg == 'median':
        agg_ser = concat.median(axis=1)
    else:
        agg_ser = concat.mean(axis=1)
    std_ser = concat.std(axis=1)
    return agg_ser.sort_index(), std_ser.sort_index()


def plot_dataset(dataset, agg='mean', marker_step=10, metric='clauses_per_second'):
    data = find_summary_and_epoch(dataset)
    # Prepare figure: 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Results: {dataset}")

    # Accuracy per epoch (top-left) - aggregate across runs
    ax = axs[0, 0]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plotted = 0
    for method, (s, e_dfs) in data.items():
        if e_dfs:
            agg_ser, std_ser = aggregate_epoch_dfs(e_dfs, agg=agg)
            if agg_ser is not None:
                # plot as dots connected by line
                # markevery controls marker density; if marker_step==0 show all markers
                me = None if marker_step == 0 else marker_step
                c = color_for(method)
                ax.plot(agg_ser.index, agg_ser.values, marker='o', linestyle='-', label=f"{method} ({agg})", markevery=me, markersize=4, color=c)
                # shaded std
                ax.fill_between(agg_ser.index, (agg_ser - std_ser).values, (agg_ser + std_ser).values, alpha=0.2, color=c)
                plotted += 1
    if plotted:
        ax.set_title('Accuracy per epoch (aggregated)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No epoch logs found', ha='center')
        ax.set_axis_off()

    # Bar: Test accuracy comparison (top-right) - compute aggregated summary accuracy if multiple rows
    ax = axs[0, 1]
    methods = []
    accuracies = []
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame):
            # try common column names
            for key in ['Accuracy on test data', 'Accuracy_on_test_data', 'acc_test', 'Accuracy']:
                if key in s.columns:
                    if agg == 'median':
                        acc = s[key].median()
                    else:
                        acc = s[key].mean()
                    methods.append(method)
                    accuracies.append(float(acc))
                    break
    if methods:
        colors = [color_for(m) for m in methods]
        ax.bar(methods, accuracies, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Test accuracy')
        ax.set_title('Test accuracy: MATLAB vs Python')
    else:
        ax.text(0.5, 0.5, 'No summary accuracy data found', ha='center')
        ax.set_axis_off()

    # Bar: Training time comparison (bottom-left) - aggregate across summary rows
    ax = axs[1, 0]
    methods = []
    times = []
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame):
            for key in ['Time', 'time', 'Duration']:
                if key in s.columns:
                    if agg == 'median':
                        tm = s[key].median()
                    else:
                        tm = s[key].mean()
                    try:
                        tm = float(tm)
                    except Exception:
                        tm = np.nan
                    methods.append(method)
                    times.append(tm)
                    break
    if methods:
        colors = [color_for(m) for m in methods]
        ax.bar(methods, times, color=colors)
        ax.set_ylabel('Time (s)')
        ax.set_title('Training time')
    else:
        ax.text(0.5, 0.5, 'No time data found', ha='center')
        ax.set_axis_off()

    # Bottom-right: combined accuracy & time table/list
    ax = axs[1, 1]
    rows = []
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame):
            # aggregate summary
            acc = None
            for key in ['Accuracy on test data', 'Accuracy_on_test_data', 'acc_test', 'Accuracy']:
                if key in s.columns:
                    acc = s[key].median() if agg == 'median' else s[key].mean()
                    break
            t = None
            for key in ['Time', 'time', 'Duration']:
                if key in s.columns:
                    t = s[key].median() if agg == 'median' else s[key].mean()
                    break
            rows.append((method, acc, t))
    if rows:
        table_data = [[r[0], f"{float(r[1]):.4f}" if (r[1] is not None and not np.isnan(r[1])) else '-', f"{float(r[2]):.4f}" if (r[2] is not None and not np.isnan(r[2])) else '-'] for r in rows]
        ax.axis('off')
        table = ax.table(cellText=table_data, colLabels=['Method','Test Acc','Time (s)'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        ax.set_title('Summary')
    else:
        ax.text(0.5, 0.5, 'No summary data found', ha='center')
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # per-dataset plots live in dataset-specific folder
    dataset_plots_dir = os.path.join(ROOT, 'result', dataset, 'plots')
    os.makedirs(dataset_plots_dir, exist_ok=True)
    outpath = os.path.join(dataset_plots_dir, f"{dataset}_comparison.png")
    fig.savefig(outpath)
    print(f"Saved plot: {outpath}")
    # Also create a compact per-dataset aggregate figure (accuracy & time) comparing methods
    # Gather methods data
    methods = []
    accs = []
    times = []
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame) and not s.empty:
            # get aggregated accuracy
            acc = None
            for key in ['Accuracy on test data', 'Accuracy_on_test_data', 'acc_test', 'Accuracy']:
                if key in s.columns:
                    acc = s[key].median() if agg == 'median' else s[key].mean()
                    break
            t = None
            for key in ['Time', 'time', 'Duration']:
                if key in s.columns:
                    t = s[key].median() if agg == 'median' else s[key].mean()
                    break
            try:
                acc = float(acc)
            except Exception:
                acc = np.nan
            try:
                t = float(t)
            except Exception:
                t = np.nan
            methods.append(method)
            accs.append(acc)
            times.append(t)
    if methods:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
        # Accuracy
        width = 0.4
        colors = [color_for(m) for m in methods]
        ax2[0].bar(methods, accs, width=width, color=colors[:len(methods)])
        ax2[0].set_ylim(0, 1)
        ax2[0].set_title(f'{dataset}: Test accuracy')
        # Time
        colors = [color_for(m) for m in methods]
        ax2[1].bar(methods, times, width=width, color=colors[:len(methods)])
        ax2[1].set_title(f'{dataset}: Training time (s)')
        outpath2 = os.path.join(dataset_plots_dir, f"{dataset}_method_compare.png")
        fig2.tight_layout()
        fig2.savefig(outpath2)
        print(f"Saved plot: {outpath2}")
    # Accuracy vs number_of_clauses per method (from summary CSVs)
    clause_methods = []
    clause_series = {}
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame) and not s.empty and 'number_of_clauses' in s.columns:
            # find a column for test accuracy
            acc_col = None
            for key in ['Accuracy on test data', 'Accuracy_on_test_data', 'acc_test', 'Accuracy']:
                if key in s.columns:
                    acc_col = key
                    break
            if acc_col is None:
                continue
            # group by number_of_clauses and aggregate
            grp = s.groupby('number_of_clauses')[acc_col]
            if agg == 'median':
                agg_grp = grp.median()
            else:
                agg_grp = grp.mean()
            # sort by clauses
            agg_grp = agg_grp.sort_index()
            clause_methods.append(method)
            clause_series[method] = agg_grp
    if clause_methods:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        for method in clause_methods:
            ser = clause_series[method]
            ax3.plot(ser.index, ser.values, marker='o', linestyle='-', label=method, color=color_for(method))
        ax3.set_xlabel('Number of clauses')
        ax3.set_ylabel('Test accuracy')
        ax3.set_title(f'{dataset}: Accuracy vs number_of_clauses')
        ax3.set_ylim(0, 1.2)
        ax3.legend()
        outpath3 = os.path.join(dataset_plots_dir, f"{dataset}_accuracy_vs_clauses.png")
        fig3.tight_layout()
        fig3.savefig(outpath3)
        print(f"Saved plot: {outpath3}")
    # Clause/time metric: support 'clauses_per_second' (clauses / s) or 'seconds_per_clause' (s / clause)
    metric_methods = []
    metric_series = {}
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame) and not s.empty and 'number_of_clauses' in s.columns and any(k in s.columns for k in ['Time', 'time', 'Duration']):
            # pick the time column name present
            time_col = None
            for key in ['Time', 'time', 'Duration']:
                if key in s.columns:
                    time_col = key
                    break
            if time_col is None:
                continue
            try:
                clauses = pd.to_numeric(s['number_of_clauses'], errors='coerce')
                times = pd.to_numeric(s[time_col], errors='coerce')
                if metric == 'clauses_per_second':
                    vals = (clauses / times).replace([np.inf, -np.inf], np.nan).dropna()
                else:
                    # seconds per clause
                    vals = (times / clauses).replace([np.inf, -np.inf], np.nan).dropna()
            except Exception:
                continue
            df_metric = s[['number_of_clauses']].copy()
            df_metric['val'] = vals
            grp = df_metric.groupby('number_of_clauses')['val']
            grp_val = grp.median() if agg == 'median' else grp.mean()
            grp_val = grp_val.sort_index()
            metric_methods.append(method)
            metric_series[method] = grp_val
    if metric_methods:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        for method in metric_methods:
            ser = metric_series[method]
            ax4.plot(ser.index, ser.values, marker='o', linestyle='-', label=method, color=color_for(method))
        ax4.set_xlabel('Number of clauses')
        if metric == 'clauses_per_second':
            ax4.set_ylabel('Clauses per second (clauses / s)')
            title_metric = 'Clauses per second'
            out_fname = f"{dataset}_clauses_per_second.png"
        else:
            ax4.set_ylabel('Seconds per clause (s)')
            title_metric = 'Seconds per clause'
            out_fname = f"{dataset}_seconds_per_clause.png"
        ax4.set_title(f'{dataset}: {title_metric}')
        ax4.legend()
        outpath4 = os.path.join(dataset_plots_dir, out_fname)
        fig4.tight_layout()
        fig4.savefig(outpath4)
        print(f"Saved plot: {outpath4}")

    # Time vs number_of_clauses per method (aggregate time by clauses)
    time_methods = []
    time_series = {}
    for method, (s, e_dfs) in data.items():
        if isinstance(s, pd.DataFrame) and not s.empty and 'number_of_clauses' in s.columns and any(k in s.columns for k in ['Time', 'time', 'Duration']):
            time_col = None
            for key in ['Time', 'time', 'Duration']:
                if key in s.columns:
                    time_col = key
                    break
            if time_col is None:
                continue
            df_mt = s[['number_of_clauses', time_col]].copy()
            df_mt['number_of_clauses'] = pd.to_numeric(df_mt['number_of_clauses'], errors='coerce')
            df_mt[time_col] = pd.to_numeric(df_mt[time_col], errors='coerce')
            df_mt = df_mt.dropna()
            if df_mt.empty:
                continue
            grp = df_mt.groupby('number_of_clauses')[time_col]
            agg_grp = grp.median() if agg == 'median' else grp.mean()
            agg_grp = agg_grp.sort_index()
            time_methods.append(method)
            time_series[method] = agg_grp
    if time_methods:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.xaxis.set_major_locator(MaxNLocator(integer=True))
        for method in time_methods:
            ser = time_series[method]
            ax5.plot(ser.index, ser.values, marker='o', linestyle='-', label=method, color=color_for(method))
        ax5.set_xlabel('Number of clauses')
        ax5.set_ylabel('Time (s)')
        ax5.set_title(f'{dataset}: Time vs number_of_clauses')
        ax5.legend()
        outpath5 = os.path.join(dataset_plots_dir, f"{dataset}_time_vs_clauses.png")
        fig5.tight_layout()
        fig5.savefig(outpath5)
        print(f"Saved plot: {outpath5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agg', choices=['mean', 'median'], default='mean', help='Aggregation method across runs')
    parser.add_argument('--methods', nargs='+', default=['MATLAB', 'Python-pure'], help='Methods to include in plots (e.g. MATLAB "Python-master" "Python-pure")')
    parser.add_argument('--marker-step', type=int, default=10, help='Show markers every N points on epoch plots (use 0 to show all)')
    parser.add_argument('--metric', choices=['clauses_per_second', 'seconds_per_clause'], default='clauses_per_second', help="Which metric to plot for the clause/time chart: 'clauses_per_second' (clauses / s) or 'seconds_per_clause' (s / clause)")
    args = parser.parse_args()

    datasets = ['mnist', 'noisy_xor', 'normal_xor']
    for ds in datasets:
        plot_dataset(ds, agg=args.agg, marker_step=args.marker_step, metric=args.metric)

    # Additionally, create an aggregated comparison across datasets for MATLAB vs Python-master (accuracy & time)
    methods = ['MATLAB', 'Python-pure']
    acc_rows = {m: [] for m in methods}
    time_rows = {m: [] for m in methods}
    labels = []
    for ds in datasets:
        labels.append(ds)
        data = find_summary_and_epoch(ds)
        for m in methods:
            s, e = data.get(m, (None, None))
            # s is a DataFrame (possibly multiple runs) or None
            if isinstance(s, pd.DataFrame) and not s.empty:
                # aggregate accuracy
                acc = None
                for key in ['Accuracy on test data', 'Accuracy_on_test_data', 'acc_test', 'Accuracy']:
                    if key in s.columns:
                        acc = s[key].median() if args.agg == 'median' else s[key].mean()
                        break
                # aggregate time
                t = None
                for key in ['Time', 'time', 'Duration']:
                    if key in s.columns:
                        t = s[key].median() if args.agg == 'median' else s[key].mean()
                        break
                try:
                    acc = float(acc)
                except Exception:
                    acc = np.nan
                try:
                    t = float(t)
                except Exception:
                    t = np.nan
            else:
                acc = np.nan
                t = np.nan
            acc_rows[m].append(acc)
            time_rows[m].append(t)

    # Plot aggregated accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    width = 0.12
    # limit to requested methods
    methods_to_plot = [m for m in methods if m in args.methods]
    for i, m in enumerate(methods_to_plot):
        ax.bar(x + (i-1)*width, acc_rows[m], width, label=m, color=color_for(m))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Test accuracy')
    ax.set_title('Aggregate test accuracy across datasets')
    ax.legend()
    outpath = os.path.join(PLOTS_DIR, 'aggregate_accuracy.png')
    fig.savefig(outpath)
    print(f"Saved plot: {outpath}")

    # Plot aggregated time
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods_to_plot):
        ax.bar(x + (i-1)*width, time_rows[m], width, label=m, color=color_for(m))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Time (s)')
    ax.set_title('Aggregate training time across datasets')
    ax.legend()
    outpath = os.path.join(PLOTS_DIR, 'aggregate_time.png')
    fig.savefig(outpath)
    print(f"Saved plot: {outpath}")


if __name__ == '__main__':
    main()

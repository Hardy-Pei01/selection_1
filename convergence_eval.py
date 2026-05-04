"""Plot convergence curves across runs in a data directory.

MOEA convergences use columns (`nfe`, `epsilon_progress`) and live in files
named `convergences_*.csv` (note the trailing 's' — plural, written by EMA
Workbench's optimize/robust_optimize).

PQL convergences use columns (`timestep`, `hypervolume`) and live in files
named `convergence_*.csv` (singular — written by morl_single / morl_moro).

This script auto-detects which kind it has and plots accordingly.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

depth = '10_2'
data_dir = f"./tree_data/moea_observable"
out_dir = "./tree_figures"


def find_convergence_file(folder_path: str):
    """Find the convergence CSV in a folder. Prefers `convergences_*` (MOEA)
    over `convergence_*` (PQL); falls back to either if only one is present.
    Returns the absolute path or None."""
    if not os.path.isdir(folder_path):
        return None
    files = os.listdir(folder_path)
    moea = sorted(f for f in files
                  if f.startswith('convergences_') and f.endswith('.csv'))
    morl = sorted(f for f in files
                  if f.startswith('convergence_') and f.endswith('.csv'))
    candidates = moea or morl
    if not candidates:
        return None
    # If multiple (e.g. one per reference scenario), take the last by name.
    return os.path.join(folder_path, candidates[-1])


def detect_columns(df: pd.DataFrame):
    """Return (x_col, y_col, x_label, y_label) for the convergence data
    in df. Raises ValueError if neither known schema matches."""
    if {'nfe', 'epsilon_progress'}.issubset(df.columns):
        return 'nfe', 'epsilon_progress', 'NFE', 'Epsilon Progress'
    if {'timestep', 'hypervolume'}.issubset(df.columns):
        return 'timestep', 'hypervolume', 'Timestep', 'Hypervolume'
    raise ValueError(
        f'Unknown convergence schema. Columns: {list(df.columns)}. '
        f'Expected (nfe, epsilon_progress) for MOEA or '
        f'(timestep, hypervolume) for PQL.'
    )


if not os.path.isdir(data_dir):
    sys.exit(f'data_dir does not exist: {os.path.abspath(data_dir)}')

os.makedirs(out_dir, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
n_plotted = 0
y_label = None

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    csv_path = find_convergence_file(folder_path)
    if csv_path is None:
        continue
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    try:
        x_col, y_col, x_label, y_label = detect_columns(df)
    except ValueError as e:
        print(f'  SKIP: {folder_name}: {e}')
        continue
    ax.plot(df[x_col], df[y_col], label=folder_name)
    n_plotted += 1

if n_plotted == 0:
    sys.exit(
        f'No convergence data found under {os.path.abspath(data_dir)}. '
        f'Expected `convergences_*.csv` (MOEA) or `convergence_*.csv` (PQL) '
        f'inside each subfolder.'
    )

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.legend(fontsize=8, loc='best')
ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.7)
ax.spines[['top', 'right']].set_visible(False)

out_path = f'{out_dir}/convergence_{depth}.png'
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved {out_path} ({n_plotted} curves)')
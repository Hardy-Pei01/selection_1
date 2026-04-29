import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from morl_baselines.common.performance_indicators import hypervolume

# ── Configuration ─────────────────────────────────────────────────────────────
data_dir = "./tree_data/moea_observable"
out_dir  = "./tree_figures"
N_OBJS   = [2, 6]

# Three regimes to compare. Order = bar order within each algorithm group.
# Tuple: (setting_in_folder, regime_in_folder, display_label, bar_color)
CONDITIONS = [
    ("intertemporal", "observable",     "Static, Observable",       "#378ADD"),
    ("table",         "observable",     "Adaptive, Observable",     "#1D9E75"),
    ("table",         "non_observable", "Adaptive, Non-observable", "#D85A30"),
]

ALGOS       = ["IBEA", "NSGAII", "MOEAD"]
ALGO_LABELS = {"IBEA": "IBEA", "NSGAII": "NSGA-II", "MOEAD": "MOEA/D"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_folder(name: str):
    """
    Folders are named `{setting}_{ALGO}_single_{n_obj}_{regime}` where
    `regime` may itself contain underscores (e.g. `non_observable`).
    Returns (setting, algo, n_obj, regime) or None if the name doesn't fit.
    """
    parts = name.split("_")
    if len(parts) < 5 or parts[2] != "single":
        return None
    setting, algo = parts[0], parts[1]
    try:
        n_obj = int(parts[3])
    except ValueError:
        return None
    regime = "_".join(parts[4:])
    return setting, algo, n_obj, regime


def find_archive_csv(folder_path: str):
    """Pick `archives_..._0.csv` (sorts before `..._combined.csv`)."""
    csv_files = [f for f in os.listdir(folder_path)
                 if f.startswith("archives_") and f.endswith(".csv")]
    if not csv_files:
        return None
    return os.path.join(folder_path, sorted(csv_files)[0])


def get_objectives(df: pd.DataFrame) -> np.ndarray:
    obj_cols = [c for c in df.columns if c.startswith("o")]
    if not obj_cols:
        obj_cols = [c for c in df.columns if c.startswith("p20_o")]
    return df[obj_cols].values.astype(float)


# ── Collect ───────────────────────────────────────────────────────────────────

# records keyed by (setting, algo, n_obj, regime) for easy lookup
records = {}

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    parsed = parse_folder(folder_name)
    if parsed is None:
        continue
    setting, algo, n_obj, regime = parsed

    csv_path = find_archive_csv(folder_path)
    if csv_path is None:
        continue
    df = pd.read_csv(csv_path)
    objectives = np.abs(get_objectives(df))
    if objectives.ndim != 2 or objectives.size == 0:
        continue

    ref_point = np.full(objectives.shape[1], -1.0)
    hv_val = hypervolume(ref_point, [tuple(r) for r in objectives])

    records[(setting, algo, n_obj, regime)] = {
        "num_solutions": len(objectives),
        "hypervolume":   hv_val,
    }


# ── Plot: one figure per n_obj, HV + archive size side by side ───────────────

def plot_for_n_obj(n_obj: int):
    fig, (ax_hv, ax_n) = plt.subplots(1, 2, figsize=(11, 5))

    n_cond  = len(CONDITIONS)
    bar_w   = 0.8 / n_cond
    centers = np.arange(len(ALGOS))

    for ci, (setting, regime, _label, color) in enumerate(CONDITIONS):
        offset = (ci - (n_cond - 1) / 2) * bar_w
        hvs, nums = [], []
        for algo in ALGOS:
            rec = records.get((setting, algo, n_obj, regime))
            hvs.append(rec["hypervolume"]   if rec else np.nan)
            nums.append(rec["num_solutions"] if rec else np.nan)

        for ax, vals, fmt in [(ax_hv, hvs, ".2f"), (ax_n, nums, "d")]:
            ax.bar(centers + offset, vals, width=bar_w, color=color,
                   edgecolor="white", linewidth=0.6)
            for x, v in zip(centers + offset, vals):
                if np.isnan(v):
                    continue
                ax.text(x, v, f"{v:{fmt}}", ha="center", va="bottom",
                        fontsize=8, color="#333")

    for ax, ylabel, title in [
        (ax_hv, "Hypervolume",
                f"Hypervolume — {n_obj}-objective"),
        (ax_n,  "Number of Non-dominated Solutions",
                f"Archive size — {n_obj}-objective"),
    ]:
        ax.set_xticks(centers)
        ax.set_xticklabels([ALGO_LABELS[a] for a in ALGOS], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        # headroom so the value labels don't clip
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)

    handles = [Patch(facecolor=c, label=l) for _, _, l, c in CONDITIONS]
    fig.legend(handles=handles, loc="lower center", ncol=len(CONDITIONS),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    fig.tight_layout()
    return fig


os.makedirs(out_dir, exist_ok=True)
for n_obj in N_OBJS:
    fig = plot_for_n_obj(n_obj)
    out = f"{out_dir}/observable_compare_{n_obj}obj.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out}")
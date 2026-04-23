import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from pymoo.indicators.hv import HV

# ── Configuration ─────────────────────────────────────────────────────────────
depth = 'lake_robust_1'
data_dir = f"./lake_data/data_{depth}"
out_dir = "./lake_figures"

# Reference points — must be strictly worse than all achievable objectives.
# Fruit tree: objectives in [-1, 0]
REF_POINTS = {
    2: np.full(2, -1.0),  # fruit tree 2-obj
    6: np.full(6, -1.0),  # lake 6-obj
    8: np.full(8, -1.0),  # fruit tree 8-obj
    14: np.full(14, -1.0),  # fruit tree 14-obj
}


# ── Label helpers ─────────────────────────────────────────────────────────────

def make_label(folder_name: str) -> str:
    parts = folder_name.split("_")

    policy_map = {
        "intertemporal": "Static",
        "dps": "DPS",
        "table": "Adaptive",
        "pareto": "Pareto",
        "indicator": "Indicator",
        "decomposition": "Decomp.",
    }
    algo_map = {"NSGAII": "NSGA-II", "MOEAD": "MOEA/D", "IBEA": "IBEA"}

    policy = policy_map.get(parts[0], parts[0].capitalize())

    algo = ""
    for p in parts:
        if p in algo_map:
            algo = algo_map[p]
            break

    return f"{policy}\n{algo}" if algo else policy


def get_obj_type(folder_name: str) -> str:
    if "many_obj" in folder_name:
        return "many_obj"
    return "multi_obj"


def get_n_obj(df: pd.DataFrame) -> int:
    obj_cols = [c for c in df.columns if c.startswith('o')]
    if not obj_cols:
        # MORL output: p20_o* columns or plain numeric
        obj_cols = [c for c in df.columns
                    if c.startswith('p20_o') or c.startswith('o')]
    return len(obj_cols)


def get_objectives(df: pd.DataFrame) -> np.ndarray:
    obj_cols = [c for c in df.columns if c.startswith('o')]
    if not obj_cols:
        obj_cols = [c for c in df.columns if c.startswith('p20_o')]
    return df[obj_cols].values.astype(float)


# ── Collect data ───────────────────────────────────────────────────────────────

records = []

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    csv_files = [f for f in sorted(os.listdir(folder_path))
                 if f.endswith(".csv") and "archive" in f.lower() or
                 f.endswith(".csv") and f.startswith("pcs")]
    # Fallback: any CSV
    if not csv_files:
        csv_files = [f for f in sorted(os.listdir(folder_path))
                     if f.endswith(".csv")]
    if not csv_files:
        continue

    df = pd.read_csv(os.path.join(folder_path, csv_files[0]))
    objectives = get_objectives(df)
    n_obj = objectives.shape[1] if objectives.ndim == 2 else 0

    if n_obj == 0 or len(objectives) == 0:
        continue

    ref_point = np.full(n_obj, -1.0)
    hv_val = float(HV(ref_point=ref_point)(objectives))

    records.append({
        "folder_name": folder_name,
        "display_label": make_label(folder_name),
        "num_solutions": len(objectives),
        "hypervolume": hv_val,
        "obj_type": get_obj_type(folder_name),
    })

# ── Sort: multi_obj group first, then many_obj ────────────────────────────────

records_multi = [r for r in records if r["obj_type"] == "multi_obj"]
records_many = [r for r in records if r["obj_type"] == "many_obj"]
sorted_records = records_multi + records_many

if not sorted_records:
    print("No data found.")
    raise SystemExit

display_labels = [r["display_label"] for r in sorted_records]
hv_values = [r["hypervolume"] for r in sorted_records]
num_solutions = [r["num_solutions"] for r in sorted_records]
obj_types = [r["obj_type"] for r in sorted_records]

# ── Bar positions with gap between groups ─────────────────────────────────────

GROUP_GAP = 1.5
positions = []
pos = 0.0
for i, obj_type in enumerate(obj_types):
    if i > 0 and obj_type != obj_types[i - 1]:
        pos += GROUP_GAP
    positions.append(pos)
    pos += 1.0

COLORS = {"multi_obj": "steelblue", "many_obj": "#e07b39"}
bar_colors = [COLORS[o] for o in obj_types]

GROUP_LABELS = {"multi_obj": "2-objective", "many_obj": "Many-objective"}

# ── Plot: HV (primary) + archive size (secondary) ─────────────────────────────

fig, axes = plt.subplots(
    1, 2,
    figsize=(max(10, len(sorted_records) * 1.1 + GROUP_GAP + 2), 6)
)

for ax, values, ylabel, title, fmt in zip(
        axes,
        [hv_values, num_solutions],
        ["Hypervolume", "Number of Non-dominated Solutions"],
        ["Hypervolume per Run (primary metric)",
         "Archive Size per Run (diagnostic)"],
        [".2f", "d"],
):
    bars = ax.bar(positions, values, color=bar_colors,
                  edgecolor="white", linewidth=0.6)

    max_val = max(values) if max(values) > 0 else 1.0
    for bar, val in zip(bars, values):
        label = f"{val:{fmt}}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            label,
            ha="center", va="bottom", fontsize=8, color="#333333"
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max_val * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Group labels below tick labels
    for grp_key, grp_label in GROUP_LABELS.items():
        grp_positions = [p for p, o in zip(positions, obj_types) if o == grp_key]
        if not grp_positions:
            continue
        mid = (grp_positions[0] + grp_positions[-1]) / 2
        ax.annotate(
            grp_label,
            xy=(mid, -0.22), xycoords=("data", "axes fraction"),
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=COLORS[grp_key],
        )

    ax.legend(
        handles=[Patch(facecolor=COLORS[k], label=GROUP_LABELS[k])
                 for k in COLORS if any(o == k for o in obj_types)],
        loc="upper right", framealpha=0.7, fontsize=9,
    )

plt.tight_layout()
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/{depth}.png", dpi=150, bbox_inches="tight")
plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

depth = 'lake_robust_1'
data_dir = f"./lake_data/data_{depth}"


def make_label(folder_name: str) -> str:
    parts = folder_name.split("_")

    context_map = {"intertemporal": "Static", "table": "Adaptive"}
    context = context_map.get(parts[0], parts[0].capitalize())

    algo_map = {"NSGAII": "NSGA-II", "MOEAD": "MOEA/D", "IBEA": "IBEA"}
    algo = algo_map.get(parts[1], parts[1])

    obj_type = "2-objective"
    for i, p in enumerate(parts):
        if p == "obj" and i > 0:
            obj_type = "8-objective" if parts[i - 1] == "many" else "2-objective"
            break

    return f"{obj_type}\n{algo}\n{context}"


def get_obj_type(folder_name: str) -> str:
    parts = folder_name.split("_")
    for i, p in enumerate(parts):
        if p == "obj" and i > 0:
            return "8-objective" if parts[i - 1] == "many" else "2-objective"
    return "2-objective"


# --- Collect data ---
records = []
for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue
    csv_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".csv")]
    if not csv_files:
        continue

    df = pd.read_csv(os.path.join(folder_path, csv_files[0]))

    records.append({
        "folder_name":   folder_name,
        "display_label": make_label(folder_name),
        "num_solutions": len(df),
        "obj_type":      get_obj_type(folder_name),
    })

# --- Sort: 2-objective group first, then 8-objective ---
records_2 = [r for r in records if r["obj_type"] == "2-objective"]
records_8 = [r for r in records if r["obj_type"] == "8-objective"]
sorted_records = records_2 + records_8

display_labels = [r["display_label"]  for r in sorted_records]
num_solutions  = [r["num_solutions"]  for r in sorted_records]
obj_types      = [r["obj_type"]       for r in sorted_records]

# --- Bar positions with a gap between groups ---
GROUP_GAP = 1.5
positions = []
pos = 0.0
for i, obj_type in enumerate(obj_types):
    if i > 0 and obj_type != obj_types[i - 1]:
        pos += GROUP_GAP
    positions.append(pos)
    pos += 1.0

# --- Plot ---
COLORS = {"2-objective": "steelblue", "8-objective": "#e07b39"}
bar_colors = [COLORS[o] for o in obj_types]

fig, ax = plt.subplots(figsize=(max(8, len(sorted_records) * 1.1 + GROUP_GAP), 6))

bars = ax.bar(positions, num_solutions, color=bar_colors, edgecolor="white", linewidth=0.6)

# Value labels on top of each bar
for bar, val in zip(bars, num_solutions):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(num_solutions) * 0.01,
        str(val),
        ha="center", va="bottom", fontsize=9, color="#333333"
    )

ax.set_xticks(positions)
ax.set_xticklabels(display_labels, rotation=0, ha="center", fontsize=8.5)
ax.set_ylabel("Number of Optimal Solutions", fontsize=11)
ax.set_title("Number of Pareto-Optimal Solutions per Run", fontsize=13, fontweight="bold")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_ylim(0, max(num_solutions) * 1.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Group labels below the tick labels
group_pos = {
    "2-objective": [p for p, o in zip(positions, obj_types) if o == "2-objective"],
    "8-objective": [p for p, o in zip(positions, obj_types) if o == "8-objective"],
}
for label, pos_list in group_pos.items():
    mid = (pos_list[0] + pos_list[-1]) / 2
    ax.annotate(
        label,
        xy=(mid, -0.24), xycoords=("data", "axes fraction"),
        ha="center", va="top", fontsize=10, fontweight="bold",
        color=COLORS[label],
    )

# Legend
ax.legend(
    handles=[Patch(facecolor=COLORS[k], label=k) for k in COLORS],
    loc="upper right", framealpha=0.7,
)

plt.tight_layout()
plt.savefig(f"./lake_figures/{depth}.png", dpi=150, bbox_inches="tight")
plt.show()
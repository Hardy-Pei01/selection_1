import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymoo.indicators.hv import HV

data_dir = "./data1"

ref_point_2 = np.array([-10]*2)
ref_point_6 = np.array([-10]*6)

folder_names = []
num_solutions = []
hv_values = []

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    files = sorted(os.listdir(folder_path))
    if not files:
        continue

    archive_file = os.path.join(folder_path, files[0])
    df = pd.read_csv(archive_file)

    obj_cols = [c for c in df.columns if c.startswith('o')]
    objectives = df[obj_cols].values

    ref_point = ref_point_6 if len(obj_cols) == 6 else ref_point_2
    hv = HV(ref_point=ref_point)

    folder_names.append(folder_name)
    num_solutions.append(len(objectives))
    # hv_values.append(hv(objectives))

    # print(f"{folder_name}: HV={hv(objectives):.4f}, Solutions={len(objectives)}")

# --- Plot ---
fig, ax = plt.subplots(figsize=(max(8, len(folder_names) * 0.6), 5))

x = range(len(folder_names))
bars = ax.bar(x, num_solutions, color="steelblue", edgecolor="white", linewidth=0.6)

# Value labels on top of each bar
for bar, val in zip(bars, num_solutions):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(num_solutions) * 0.01,
        str(val),
        ha="center", va="bottom", fontsize=9, color="#333333"
    )

ax.set_xticks(list(x))
ax.set_xticklabels(folder_names, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Number of Optimal Solutions", fontsize=11)
ax.set_xlabel("Folder / Run", fontsize=11)
ax.set_title("Number of Pareto-Optimal Solutions per Run", fontsize=13, fontweight="bold")
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_ylim(0, max(num_solutions) * 1.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
# plt.savefig("multi_uncertain.png", dpi=150, bbox_inches="tight")
plt.show()


# A line of each algorithm performance with the depth increasing (2-objective and 8-objective)
# A line of ratio with the depth increasing
# Path dependent fruit tree

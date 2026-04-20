import os
import pandas as pd
import matplotlib.pyplot as plt

depth = 'lake_robust_1'
data_dir = f"./lake_data/data_{depth}"

for folder_name in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    files = sorted(os.listdir(folder_path))
    if not files:
        continue
    csv_files = [f for f in files if f.endswith(".csv")]

    convergence_file = os.path.join(folder_path, csv_files[-1])
    df = pd.read_csv(convergence_file)

    plt.plot(df["nfe"], df["epsilon_progress"], label=folder_name)

plt.xlabel("NFE")
plt.ylabel("Epsilon Progress")
plt.legend()
plt.show()
plt.savefig(f"./lake_figures/convergence_{depth}.png", dpi=150, bbox_inches="tight")

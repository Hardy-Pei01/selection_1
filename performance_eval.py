import os
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

data_dir = "./moea/data"

# Reference point — must dominate all solutions.
# Since rewards are in [0, 10] and you're minimising, 10 is a safe upper bound.
ref_point_2 = np.array([10.0, 10.0])
ref_point_6 = np.array([10.0] * 6)

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

    print(f"{folder_name}: {hv(objectives):.4f}")
    print(f"Number of optimal solutions: {len(objectives)}")
    print()
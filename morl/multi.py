"""MORL multi-scenario runner — mirrors moea/multi.py."""

import os
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.moro import run_morl, _get_depth

# Reference slip_prob values — mirrors multi_tree_params.references
# in moea/method_config.py: four explicit references + default scenario.
REFERENCE_SLIP_PROBS = [0.05, 0.1, 0.15, 0.2, 0.0]


def run_multi(scoring, timesteps, ref_point, n_obj, csv_path,
              num_weight_divisions, neighbourhood_size,
              output_folder, file_end):
    """MORL analog of moea/multi.py run_moea.

    Trains one PQL agent per reference slip_prob, saves each archive
    individually, then merges into a combined PCS — mirrors moea_multi
    running one optimisation per reference scenario and collecting archives.
    """
    depth = _get_depth(csv_path)
    all_pcs = []

    for ref_num, slip_prob in enumerate(REFERENCE_SLIP_PROBS):
        ref_file_end = f'{file_end}_{ref_num}'
        print(f'  Reference scenario {ref_num} (slip_prob={slip_prob:.2f})')

        env = FruitTreeEnv(
            depth=depth, reward_dim=n_obj,
            csv_path=csv_path, observe=True,
            slip_prob=slip_prob,
        )

        pcs_df = run_morl(env, scoring, timesteps, ref_point,
                          num_weight_divisions, neighbourhood_size,
                          output_folder, ref_file_end)
        all_pcs.append(pcs_df)

    # Merge all reference archives — mirrors multi.py concatenating
    # archives collected across reference scenarios.
    if any(len(p) > 0 for p in all_pcs):
        combined = pd.concat([p for p in all_pcs if len(p) > 0],
                              ignore_index=True)
    else:
        combined = pd.DataFrame()

    combined.to_csv(f'{output_folder}/pcs_{file_end}_combined.csv', index=False)
    return combined
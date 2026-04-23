"""Fruit-tree MORL experiment runner — mirrors run_tree_moea.py.

Toggle the flag dicts below to select which experiments to run, then execute:

    python run_tree_morl.py

Correspondence with run_tree_moea.py:
  run_scoring        <-> run_evo_method   (algorithm selection)
  run_scenario_method                      (identical structure)
  obj_uncertain                            (identical structure)
  param_uncertain                          (identical structure)
  nfe_settings       <-> timestep_settings
  morl_multi()       <-> moea_multi()
  morl_moro()        <-> moea_moro()
  multi_tree_morl_params <-> multi_tree_params
  moro_tree_morl_params  <-> moro_tree_params
"""

import time
from collections import defaultdict

import numpy as np

from params_config import tree_depth, tree_multi_obj, tree_many_obj
from morl.morl_method_config import (
    multi_tree_morl_params,
    moro_tree_morl_params,
    morl_multi,
    morl_moro,
)

# ── Top-level output folder ───────────────────────────────────────────────────
root_folder = f'./data_morl_{tree_depth}'

# ── Experiment toggles ────────────────────────────────────────────────────────
# Set a value to True to include that dimension in the run grid.

# PQL action-evaluation method (analog of run_evo_method in run_tree_moea.py)
run_scoring = {
    'pareto': True,
    'indicator': True,
    'decomposition': True,
}

# Scenario method:
#   single — one run on the base scenario (slip_prob=0.0), no uncertainty
#   multi  — one run per reference slip_prob + base, results combined
#   moro   — one run across 50 sampled scenarios (robust optimisation)
run_scenario_method = {
    'single': True,
    'multi': False,
    'moro': False,
}

# Objective dimensionality
obj_uncertain = {
    'multi_obj': True,  # 2-objective tree
    'many_obj': True,  # 6-objective tree  (tree_many_obj from params_config)
}

# Parametric uncertainty (slip_prob)
param_uncertain = {
    'deterministic': True,  # fixed slip_prob=0.0  — only valid with 'single'
    'robust': False,  # uncertain slip_prob  — only valid with 'multi'/'moro'
}


# ── Timestep grid ─────────────────────────────────────────────────────────────
# Mirrors nfe_settings in run_tree_moea.py.
def _nested():
    return defaultdict(_nested)


timestep_settings = _nested()
timestep_settings['single']['multi_obj']['deterministic'] = 100000
timestep_settings['single']['many_obj']['deterministic'] = 100000
timestep_settings['multi']['multi_obj']['robust'] = 50000
timestep_settings['multi']['many_obj']['robust'] = 50000
timestep_settings['moro']['multi_obj']['robust'] = 50000
timestep_settings['moro']['many_obj']['robust'] = 50000

# ── PQL hyperparameters ───────────────────────────────────────────────────────
# Objective-count-dependent settings (no moea equivalent — PQL-specific).
num_weight_divisions = {
    'multi_obj': 149,  # C(149+2-1, 149) = 150 weight vectors for 2 objectives
    'many_obj': 5,  # manageable grid for 6 objectives
}
neighbourhood_size = 10

# ── Objective metadata ────────────────────────────────────────────────────────
ref_points = {
    'multi_obj': np.full(tree_multi_obj, -1.0),  # was -10.0
    'many_obj': np.full(tree_many_obj,  -1.0),  # was -10.0
}
csv_paths = {
    'multi_obj': f'./fruits/depth{tree_depth}_dim{tree_multi_obj}.csv',
    'many_obj': f'./fruits/depth{tree_depth}_dim{tree_many_obj}.csv',
}
num_objectives = {
    'multi_obj': tree_multi_obj,
    'many_obj': tree_many_obj,
}

# ── Main loop — mirrors run_tree_moea.py ──────────────────────────────────────
if __name__ == '__main__':

    for key_scoring, val_scoring in run_scoring.items():
        if not val_scoring:
            continue

        for key_3, val_3 in run_scenario_method.items():
            if not val_3:
                continue

            for key_obj, val_obj in obj_uncertain.items():
                if not val_obj:
                    continue

                for key_param, val_param in param_uncertain.items():
                    if not val_param:
                        continue

                    # Enforce valid (scenario_method, uncertainty) combinations —
                    # mirrors the same guards in run_tree_moea.py.
                    if key_3 == 'single' and key_param == 'robust':
                        continue  # single only makes sense without param uncertainty
                    if key_3 in ('multi', 'moro') and key_param == 'deterministic':
                        continue  # multi/moro require param uncertainty

                    name = f'{key_scoring}_{key_3}_{key_obj}_{key_param}'
                    timesteps = timestep_settings[key_3][key_obj][key_param]
                    n_obj = num_objectives[key_obj]
                    csv = csv_paths[key_obj]
                    ref = ref_points[key_obj]
                    nwd = num_weight_divisions[key_obj]

                    print('--------------------------------------------------------------------')
                    print(f'This experiment is {name}')
                    print('--------------------------------------------------------------------')

                    # robust=False for 'single' (non_param), True for 'multi'/'moro' (param)
                    robust = (key_param == 'robust')

                    start_time = time.time()

                    if key_3 == 'moro':
                        # Robust optimisation across all scenarios — mirrors moea_moro
                        params = moro_tree_morl_params(
                            name=name,
                            timesteps=timesteps,
                            scoring=key_scoring,
                            root_folder=root_folder,
                            many_obj=(key_obj == 'many_obj'),
                            robust=robust,
                            num_weight_divisions=nwd,
                            neighbourhood_size=neighbourhood_size,
                        )
                        result = morl_moro(
                            params=params,
                            ref_point=ref,
                            n_obj=n_obj,
                            csv_path=csv,
                            start_time=start_time,
                        )

                    else:
                        # Single scenario or separate-per-reference — mirrors moea_multi
                        params = multi_tree_morl_params(
                            name=name,
                            timesteps=timesteps,
                            scoring=key_scoring,
                            root_folder=root_folder,
                            many_obj=(key_obj == 'many_obj'),
                            robust=robust,
                            num_weight_divisions=nwd,
                            neighbourhood_size=neighbourhood_size,
                        )
                        result = morl_multi(
                            params=params,
                            ref_point=ref,
                            n_obj=n_obj,
                            csv_path=csv,
                            start_time=start_time,
                        )

                    elapsed = int(time.time() - start_time)
                    if isinstance(result, list):
                        total_pcs = sum(len(r) for r in result if not r.empty)
                    else:
                        total_pcs = len(result)
                    print(
                        f'  Done in {time.strftime("%H:%M:%S", time.gmtime(elapsed))}. '
                        f'Total PCS size: {total_pcs}.'
                    )

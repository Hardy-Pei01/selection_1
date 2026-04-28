import time
from collections import defaultdict

import numpy as np

from params_config import lake_multi_obj, lake_many_obj
from morl.morl_method_config import (
    multi_lake_morl_params,
    moro_lake_morl_params,
    morl_multi_lake,
    morl_moro_lake,
)

# ── Top-level output folder ───────────────────────────────────────────────────
root_folder = './data_morl_lake'

# ── Experiment toggles ────────────────────────────────────────────────────────

# PQL scoring method — analog of run_evo_method in run_lake_moea.py
run_scoring = {
    'pareto': 1,
    'indicator': 1,
    'decomposition': 1,
}

# Scenario method — mirrors run_lake_moea.py structure
# Note: DPS policy is MOEA-only (continuous RBF levers incompatible with PQL).
#       Only the intertemporal policy is supported here.
run_scenario_method = {
    'single': 1,
    'multi': 0,
    'moro': 0,
}

# Objective dimensionality
obj_uncertain = {
    'multi_obj': 1,  # 2-objective lake
    'many_obj': 1,  # 6-objective lake
}

# Parametric uncertainty — mirrors run_lake_moea.py key names
param_uncertain = {
    'deterministic': 1,  # fixed default scenario — only valid with 'single'
    'robust': 0,  # uncertain lake params — only valid with 'multi'/'moro'
}


# ── Timestep grid ─────────────────────────────────────────────────────────────

def _nested():
    return defaultdict(_nested)


timestep_settings = _nested()
timestep_settings['single']['multi_obj']['deterministic'] = 200000
timestep_settings['single']['many_obj']['deterministic'] = 200000
timestep_settings['multi']['multi_obj']['robust'] = 200000
timestep_settings['multi']['many_obj']['robust'] = 200000
timestep_settings['moro']['multi_obj']['robust'] = 200000
timestep_settings['moro']['many_obj']['robust'] = 200000

# ── PQL hyperparameters ───────────────────────────────────────────────────────

num_weight_divisions = {
    'multi_obj': 149,  # 150 weight vectors for 2 objectives
    'many_obj': 5,  # manageable grid for 6 objectives
}
neighbourhood_size = 10

# ── Objective metadata ────────────────────────────────────────────────────────
# Reference points must be strictly below all achievable reward values.
# Lake rewards: utility in [0, ~2], reliability in [0, 1], -inertia in [-1, 0].
# np.full(n_obj, -1.0) is a safe lower bound for all objectives.
ref_points = {
    'multi_obj': np.full(lake_multi_obj, -1.1),
    'many_obj': np.full(lake_many_obj, -1.1),
}
num_objectives = {
    'multi_obj': lake_multi_obj,
    'many_obj': lake_many_obj,
}

# ── Main loop ─────────────────────────────────────────────────────────────────

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
                    # mirrors the same guards in run_tree_morl.py and run_lake_moea.py.
                    if key_3 == 'single' and key_param == 'robust':
                        continue  # single only makes sense without param uncertainty
                    if key_3 in ('multi', 'moro') and key_param == 'deterministic':
                        continue  # multi/moro require param uncertainty

                    name = f'{key_scoring}_{key_3}_{key_obj}_{key_param}'
                    timesteps = timestep_settings[key_3][key_obj][key_param]
                    n_obj = num_objectives[key_obj]
                    ref = ref_points[key_obj]
                    nwd = num_weight_divisions[key_obj]
                    robust = (key_param == 'robust')

                    print('--------------------------------------------------------------------')
                    print(f'This experiment is {name}')
                    print('--------------------------------------------------------------------')

                    start_time = time.time()

                    if key_3 == 'moro':
                        params = moro_lake_morl_params(
                            name=name,
                            timesteps=timesteps,
                            scoring=key_scoring,
                            root_folder=root_folder,
                            many_obj=(key_obj == 'many_obj'),
                            robust=robust,
                            num_weight_divisions=nwd,
                            neighbourhood_size=neighbourhood_size,
                        )
                        result = morl_moro_lake(
                            params=params,
                            ref_point=ref,
                            n_obj=n_obj,
                            start_time=start_time,
                        )

                    else:
                        params = multi_lake_morl_params(
                            name=name,
                            timesteps=timesteps,
                            scoring=key_scoring,
                            root_folder=root_folder,
                            many_obj=(key_obj == 'many_obj'),
                            robust=robust,
                            num_weight_divisions=nwd,
                            neighbourhood_size=neighbourhood_size,
                        )
                        result = morl_multi_lake(
                            params=params,
                            ref_point=ref,
                            n_obj=n_obj,
                            start_time=start_time,
                        )

                    elapsed = int(time.time() - start_time)
                    if isinstance(result, list):
                        total_pcs = sum(len(r) for r in result if not r.empty)
                    else:
                        total_pcs = len(result) if result is not None else 0
                    print(
                        f'  Done in {time.strftime("%H:%M:%S", time.gmtime(elapsed))}. '
                        f'Total PCS size: {total_pcs}.'
                    )

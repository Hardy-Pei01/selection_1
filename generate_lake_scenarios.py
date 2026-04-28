"""Generate a fixed ensemble of lake scenarios for MORO training.

Saves 50 scenarios to lakes/lake_scenarios.npy as a structured numpy array.
Uncertainty bounds are read from params_config.py to stay consistent with
the MOEA uncertainty space.
"""

import os
import numpy as np
from params_config import (
    multi_objs_lake_params,
    lake_n_scenarios,
)

SCENARIO_SEED = 42


def generate_lake_scenarios(n_scenarios=lake_n_scenarios, seed=SCENARIO_SEED):
    rng = np.random.default_rng(seed)

    # Read bounds from params_config uncertainties
    bounds = {u.name: (u.lower_bound, u.upper_bound)
              for u in multi_objs_lake_params['uncertainties']}

    dt = np.dtype([
        ('b1', np.float64), ('q1', np.float64),
        ('b2', np.float64), ('q2', np.float64),
        ('inflow_seed1', np.int64), ('inflow_seed2', np.int64),
    ])
    scenarios = np.empty(n_scenarios, dtype=dt)

    scenarios['b1'] = rng.uniform(*bounds['b1'], n_scenarios)
    scenarios['q1'] = rng.uniform(*bounds['q1'], n_scenarios)
    scenarios['b2'] = rng.uniform(*bounds['b2'], n_scenarios)
    scenarios['q2'] = rng.uniform(*bounds['q2'], n_scenarios)
    scenarios['inflow_seed1'] = rng.integers(
        bounds['inflow_seed1'][0], bounds['inflow_seed1'][1] + 1, n_scenarios)
    scenarios['inflow_seed2'] = rng.integers(
        bounds['inflow_seed2'][0], bounds['inflow_seed2'][1] + 1, n_scenarios)

    return scenarios


def main():
    scenarios = generate_lake_scenarios()

    os.makedirs('lakes', exist_ok=True)
    out_path = 'lakes/lake_scenarios.npy'
    np.save(out_path, scenarios)

    print(f"Generated {len(scenarios)} lake scenarios -> {out_path}")
    for i in [0, 12, 24, 37, 49]:
        s = scenarios[i]
        print(f"  scenario {i:2d}: b1={s['b1']:.3f}, q1={s['q1']:.3f}, "
              f"b2={s['b2']:.3f}, q2={s['q2']:.3f}, "
              f"inflow_seeds=({s['inflow_seed1']},{s['inflow_seed2']})")


if __name__ == '__main__':
    main()
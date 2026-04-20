"""MORL MORO runner — mirrors moea/moro.py."""

import os
import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.pql import PQL


class MoroFruitTreeEnv(FruitTreeEnv):
    """FruitTreeEnv that trains across a fixed set of 50 slip_prob scenarios.

    At each reset(), one scenario is selected uniformly from the fixed set.
    Mirrors MOEA MORO exactly: the agent is optimised across all 50 scenarios
    simultaneously rather than a single deterministic environment.
    """

    N_SCENARIOS = 50

    def __init__(self, depth, reward_dim, csv_path, observe, seed=42):
        rng = np.random.default_rng(seed)
        self._scenarios = rng.uniform(0.0, 0.2, self.N_SCENARIOS)
        self._scenario_rng = np.random.default_rng(seed + 1)
        slip_prob = float(self._scenarios[0])
        super().__init__(depth=depth, reward_dim=reward_dim,
                         csv_path=csv_path, observe=observe,
                         slip_prob=slip_prob)

    def reset(self, *, seed=None, options=None):
        idx = self._scenario_rng.integers(self.N_SCENARIOS)
        self.slip_prob = float(self._scenarios[idx])
        return super().reset(seed=int(self.slip_prob * 1e6), options=options)


def run_morl(env, scoring, timesteps, ref_point,
             num_weight_divisions, neighbourhood_size,
             output_folder, file_end):
    """Train one PQL agent and save PCS + convergence.

    Core runner shared by moro and multi — analogous to the
    evaluator.optimize / evaluator.robust_optimize call inside
    moea/moro.py and moea/multi.py.
    """
    os.makedirs(output_folder, exist_ok=True)

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=0.8,
        initial_epsilon=1.0,
        epsilon_decay_steps=int(timesteps * 0.8),
        final_epsilon=0.05,
        num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
    )

    pcs, convergence_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 100),
    )

    if pcs:
        pcs_arr = np.array([list(v) for v in pcs])
        pcs_df = pd.DataFrame(
            pcs_arr,
            columns=[f'o{i+1}' for i in range(pcs_arr.shape[1])]
        )
    else:
        pcs_df = pd.DataFrame()

    pcs_df.to_csv(f'{output_folder}/pcs_{file_end}.csv', index=False)
    pd.DataFrame(convergence_log).to_csv(
        f'{output_folder}/convergence_{file_end}.csv', index=False)

    return pcs_df


def run_moro(scoring, timesteps, ref_point, n_obj, csv_path,
             num_weight_divisions, neighbourhood_size,
             output_folder, file_end):
    """MORL analog of moea/moro.py run_moea.

    Trains a single PQL agent whose environment cycles through 50
    pre-sampled slip_prob scenarios — equivalent to robust_optimize
    over 50 sampled scenarios.
    """
    env = MoroFruitTreeEnv(
        depth=_get_depth(csv_path), reward_dim=n_obj,
        csv_path=csv_path, observe=True,
    )
    return run_morl(env, scoring, timesteps, ref_point,
                    num_weight_divisions, neighbourhood_size,
                    output_folder, file_end)


def _get_depth(csv_path):
    """Extract tree depth from CSV filename (depth{d}_dim{n}.csv)."""
    import re
    m = re.search(r'depth(\d+)', csv_path)
    if m:
        return int(m.group(1))
    raise ValueError(f'Cannot infer depth from csv_path: {csv_path}')
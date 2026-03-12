"""PQL experiments for fruit tree — mirrors the MOEA experimental design.

Runs three action-evaluation methods (pareto_cardinality, hypervolume,
decomposition) across problem variants:

Single scenario (non-robust):
    - Optimise under the default scenario (b=0.42, q=2.0)
    - Mirrors: moea_multi with robust=False

Multi-scenario:
    - Train separately under 4 pre-defined scenarios + default (5 runs)
    - Combine Pareto fronts into a single non-dominated set
    - Mirrors: moea_multi with robust=True

MORO (robust optimisation):
    - 50 (b, q) scenarios sampled upfront; one drawn per episode
    - Mirrors: moea_moro with robust_optimize over 50 scenarios
"""

import os
import time

import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.pql import PQL
from morl_baselines.common.pareto import get_non_dominated


# ===========================================================================
# Environment wrappers
# ===========================================================================
PRESSURE = {0: 0.02, 1: 0.1}


def get_resource_pressure(x, action, b, q):
    recovery = (1 - b) * x
    tipping = x**q / (1 + x**q)
    extraction = PRESSURE[action]
    return recovery + tipping + extraction


class FruitTreeGymWrapper:
    """Wraps FruitTreeEnv with fixed (b, q) resource pressure.

    Used for both single-scenario and multi-scenario runs.
    Each instance represents one fixed scenario.
    """

    def __init__(self, depth, reward_dim, observe=True, b=0.42, q=2.0):
        self._env = FruitTreeEnv(depth=depth, reward_dim=reward_dim, observe=observe)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_space = self._env.reward_space
        self.b = b
        self.q = q
        self.x = 0.0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        state, info = self._env.reset(seed=seed)
        self.x = 0.0
        return state, info

    def step(self, action):
        state, reward, terminal = self._env.step(action)
        self.x = get_resource_pressure(self.x, action, self.b, self.q)
        degraded_reward = reward * np.exp(-self.x)
        return state, degraded_reward, terminal, False, {}


class MOROFruitTreeGymWrapper:
    """Fruit tree with MORO-style robust evaluation.

    Samples a pool of N (b, q) scenarios upfront.  On each episode reset,
    one scenario is drawn from the pool.  The agent experiences a single
    (b, q) per episode, just as the MOEA's robust_optimize evaluates each
    candidate policy under one scenario per function evaluation.  Over
    many episodes PQL's Q-sets naturally aggregate across the full
    scenario distribution.
    """

    def __init__(self, depth, reward_dim, observe=True,
                 b_range=(0.1, 0.45), q_range=(2.0, 4.5),
                 n_scenarios=50, seed=42):
        self._env = FruitTreeEnv(depth=depth, reward_dim=reward_dim, observe=observe)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.reward_space = self._env.reward_space

        # Sample scenario pool once upfront — mirrors MORO's
        # buildOptimizationScenarios(model, 50)
        rng = np.random.default_rng(seed)
        self.scenarios_b = rng.uniform(*b_range, size=n_scenarios)
        self.scenarios_q = rng.uniform(*q_range, size=n_scenarios)
        self.n_scenarios = n_scenarios

        self._rng = np.random.default_rng()
        self.b = None
        self.q = None
        self.x = 0.0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        state, info = self._env.reset(seed=seed)

        # Draw one scenario from the pool for this episode
        idx = self._rng.integers(self.n_scenarios)
        self.b = self.scenarios_b[idx]
        self.q = self.scenarios_q[idx]
        self.x = 0.0
        return state, info

    def step(self, action):
        state, reward, terminal = self._env.step(action)
        self.x = get_resource_pressure(self.x, action, self.b, self.q)
        degraded_reward = reward * np.exp(-self.x)
        return state, degraded_reward, terminal, False, {}


# ===========================================================================
# Experiment configuration — mirrors MOEA start_run.py
# ===========================================================================
DEPTH = 7
TOTAL_TIMESTEPS = 50000
ROOT_FOLDER = './data'

# Pre-defined reference scenarios — same as multi_params.references
MULTI_SCENARIOS = [
    {'b': 0.268340928, 'q': 3.502868198},
    {'b': 0.100879116, 'q': 3.699779508},
    {'b': 0.218652257, 'q': 2.050630370},
    {'b': 0.161967233, 'q': 3.868530616},
]
DEFAULT_SCENARIO = {'b': 0.42, 'q': 2.0}

run_scorer = {
    'pareto': True,
    'indicator': True,
    'decomposition': True,
}

run_objectives = {
    'multi_obj': True,
    'many_obj': True,
}

run_robustness = {
    'single': True,
    'multi': True,
    'moro': True,
}

reward_dim_map = {
    'multi_obj': 2,
    'many_obj': 6,
}

ref_point_map = {
    'multi_obj': np.array([0.0, 0.0]),
    'many_obj': np.array([0.0] * 6),
}


def train_pql(env, ref_point, scorer_name, timesteps):
    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=0.8,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps // 2,
        final_epsilon=0.1,
    )
    pcs, convergence_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scorer_name,
        log_every=1000,
    )
    return pcs, convergence_log


def save_results(pcs, convergence_log, name, output_folder, elapsed, ref_num=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    suffix = f'_{ref_num}' if ref_num is not None else ''

    # Save convergence log
    if convergence_log:
        conv_df = pd.DataFrame(convergence_log)
        conv_df.to_csv(f'{output_folder}/convergence_{name}{suffix}.csv', index=False)

    if not pcs:
        print(f"  WARNING: Empty Pareto front")
        print(f"  Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        return

    reward_dim = len(next(iter(pcs)))
    rows = []
    for vec in pcs:
        row = {f'o{i+1}': vec[i] for i in range(reward_dim)}
        if ref_num is not None:
            row['reference_scenario'] = ref_num
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{output_folder}/{name}{suffix}.csv', index=False)

    print(f"  Pareto front size: {len(pcs)}")
    print(f"  Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")


# ===========================================================================
# Main loop
# ===========================================================================
if __name__ == '__main__':

    for scorer_name, scorer_enabled in run_scorer.items():
        if not scorer_enabled:
            continue

        for obj_key, obj_enabled in run_objectives.items():
            if not obj_enabled:
                continue

            for robust_key, robust_enabled in run_robustness.items():
                if not robust_enabled:
                    continue

                reward_dim = reward_dim_map[obj_key]
                ref_point = ref_point_map[obj_key]
                name = f'{scorer_name}_{obj_key}_{robust_key}'
                output_folder = f'{ROOT_FOLDER}/{name}'

                print('=' * 68)
                print(f'This experiment is {name}')
                print('=' * 68)

                start_time = time.time()

                # -----------------------------------------------------------
                # Single scenario (non-robust)
                # Mirrors: moea_multi with robust=False → one default ref
                # -----------------------------------------------------------
                if robust_key == 'single':
                    env = FruitTreeGymWrapper(
                        depth=DEPTH, reward_dim=reward_dim,
                        **DEFAULT_SCENARIO,
                    )
                    pcs, conv = train_pql(env, ref_point, scorer_name, TOTAL_TIMESTEPS)
                    elapsed = time.time() - start_time
                    save_results(pcs, conv, name, output_folder, elapsed)

                # -----------------------------------------------------------
                # Multi-scenario
                # Mirrors: moea_multi with robust=True → 4 refs + default
                # Train PQL per scenario, combine into non-dominated set
                # -----------------------------------------------------------
                elif robust_key == 'multi':
                    all_scenarios = MULTI_SCENARIOS + [DEFAULT_SCENARIO]
                    all_vecs = set()

                    for idx, scenario in enumerate(all_scenarios):
                        print(f'  Reference scenario {idx}: '
                              f'b={scenario["b"]:.4f}, q={scenario["q"]:.4f}')

                        env = FruitTreeGymWrapper(
                            depth=DEPTH, reward_dim=reward_dim,
                            **scenario,
                        )
                        pcs, conv = train_pql(env, ref_point, scorer_name, TOTAL_TIMESTEPS)

                        elapsed = time.time() - start_time
                        save_results(pcs, conv, name, output_folder, elapsed, ref_num=idx)

                        all_vecs.update(pcs)

                    # Combine all scenario fronts into one non-dominated set
                    combined_pcs = get_non_dominated(all_vecs)
                    elapsed = time.time() - start_time
                    save_results(combined_pcs, None, f'{name}_combined',
                                 output_folder, elapsed)

                # -----------------------------------------------------------
                # MORO (robust optimisation)
                # Mirrors: moea_moro → 50 scenarios sampled upfront,
                # one scenario drawn per episode
                # -----------------------------------------------------------
                elif robust_key == 'moro':
                    env = MOROFruitTreeGymWrapper(
                        depth=DEPTH, reward_dim=reward_dim,
                        n_scenarios=50,
                    )
                    pcs, conv = train_pql(env, ref_point, scorer_name, TOTAL_TIMESTEPS)
                    elapsed = time.time() - start_time
                    save_results(pcs, conv, name, output_folder, elapsed)
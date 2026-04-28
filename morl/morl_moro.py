import os
import re
import time
import numpy as np
import pandas as pd
from fruit_tree import FruitTreeEnv
from morl.pql import PQL
from morl.policy_eval import extract_policy, \
    evaluate_policies_across_scenarios, compute_robustness
from params_config import slip_patterns_path, nd_size_cap


class MoroFruitTreeEnv(FruitTreeEnv):
    def __init__(self, depth, reward_dim, csv_path, observe,
                 patterns_path, seed=42):
        self._patterns = np.load(patterns_path)  # load once
        self._scenario_rng = np.random.default_rng(seed)
        super().__init__(depth=depth, reward_dim=reward_dim,
                         csv_path=csv_path, observe=observe,
                         scenario_index=0,
                         slip_patterns_path=patterns_path)

    def reset(self, *, seed=None, options=None):
        idx = int(self._scenario_rng.integers(len(self._patterns)))
        self._slip_pattern = self._patterns[idx]
        return super().reset(seed=seed, options=options)


def _get_depth(csv_path):
    """Extract tree depth from a CSV filename containing 'depth{d}'."""
    m = re.search(r'depth(\d+)', csv_path)
    if m:
        return int(m.group(1))
    raise ValueError(f'Cannot infer tree depth from csv_path: {csv_path}')


# ── Main runner ───────────────────────────────────────────────────────────────

def run_moro(
        scoring,
        timesteps,
        ref_point,
        n_obj,
        csv_path,
        num_weight_divisions,
        neighbourhood_size,
        output_folder,
        file_end,
        start_time=None,
):
    os.makedirs(output_folder, exist_ok=True)

    depth = _get_depth(csv_path)
    env = MoroFruitTreeEnv(
        depth=depth, reward_dim=n_obj,
        csv_path=csv_path, observe=True,
        patterns_path=slip_patterns_path,
    )

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
        final_epsilon=0.05,
        num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
        nd_update_freq=1,
        robust=True,
        max_nd_size=nd_size_cap
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 100),
    )

    # ── Build PCS dataframe ───────────────────────────────────────────────
    if pcs:
        pcs_arr = np.array([list(v) for v in pcs])
        pcs_df = pd.DataFrame(
            pcs_arr,
            columns=[f'o{i + 1}' for i in range(pcs_arr.shape[1])],
        )
    else:
        pcs_df = pd.DataFrame()

    # ── Build convergence dataframe ───────────────────────────────────────
    conv_df = pd.DataFrame(conv_log)

    # Attach elapsed wall-clock time — mirrors moea/moea_moro.py inside the
    # evaluator block where conv['time'] is set.
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    # ── Persist ───────────────────────────────────────────────────────────
    n_scenarios = len(env._patterns)

    env_factory = lambda idx: FruitTreeEnv(
        depth=depth, reward_dim=n_obj,
        csv_path=csv_path, observe=True,
        scenario_index=idx,
        slip_patterns_path=slip_patterns_path,
    )

    eval_df = evaluate_policies_across_scenarios(
        agent=agent,
        env_factory=env_factory,
        n_scenarios=n_scenarios,
    )
    robust_pcs_df = compute_robustness(eval_df, n_obj)

    # ── Persist ───────────────────────────────────────────────────────
    policy_rows = []
    for pol_id, target_vec in enumerate(agent.archive):
        decisions = extract_policy(agent, target_vec)
        row = {'policy_id': pol_id}
        row.update({f'l{i}': d for i, d in enumerate(decisions)})
        policy_rows.append(row)

    policies_df = pd.DataFrame(policy_rows) if policy_rows else pd.DataFrame()
    policies_df.to_csv(f'{output_folder}/policies_{file_end}.csv', index=False)
    robust_pcs_df.to_csv(f'{output_folder}/pcs_{file_end}.csv', index=False)
    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)

    return policies_df

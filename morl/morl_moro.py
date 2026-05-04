import os
import re
import time
import numpy as np
import pandas as pd
from fruit_tree import FruitTreeEnv
from two_lake import TwoLakeEnv
from morl.pql import PQL
from policy_eval import extract_policy, extract_lake_policy, \
    evaluate_tree_policies_across_scenarios, compute_robustness, \
    evaluate_lake_policies_across_scenarios
from params_config import slip_patterns_path, nd_size_cap_tree, \
    lake_scenarios_path, nd_size_cap_lake, nd_update_freq_tree, nd_update_freq_lake, \
    archive_cap_tree, archive_cap_lake, total_years, years_per_action, \
    gamma_tree, gamma_lake


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


class MoroLakeEnv(TwoLakeEnv):
    def __init__(self, n_obj, scenarios_path, seed=42):
        scenarios = np.load(scenarios_path)  # load once
        self._scenarios = scenarios
        self._scenario_rng = np.random.default_rng(seed)
        # Initialise with first scenario
        s = scenarios[0]
        self._current_idx = -1
        super().__init__(
            b1=float(s['b1']), q1=float(s['q1']),
            b2=float(s['b2']), q2=float(s['q2']),
            inflow_seed1=int(s['inflow_seed1']),
            inflow_seed2=int(s['inflow_seed2']),
            Pcrit1=float(s['Pcrit1']),
            Pcrit2=float(s['Pcrit2']),
            num_obj=n_obj,
            total_years=total_years,
            years_per_action=years_per_action,
        )

    def reset(self, *, seed=None, options=None):
        idx = int(self._scenario_rng.integers(len(self._scenarios)))

        # Only regenerate inflows and Pcrit when scenario actually changes
        if idx != self._current_idx:
            self._current_idx = idx
            s = self._scenarios[idx]
            self.b1, self.q1 = float(s['b1']), float(s['q1'])
            self.b2, self.q2 = float(s['b2']), float(s['q2'])
            self.inflow_seed1 = int(s['inflow_seed1'])
            self.inflow_seed2 = int(s['inflow_seed2'])
            inflow_rng1 = np.random.default_rng(self.inflow_seed1)
            self._inflows1 = inflow_rng1.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            inflow_rng2 = np.random.default_rng(self.inflow_seed2)
            self._inflows2 = inflow_rng2.lognormal(
                self._ln_mean, self._ln_sigma, size=self.total_years)
            self.Pcrit1 = float(s['Pcrit1'])
            self.Pcrit2 = float(s['Pcrit2'])

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
        gamma=gamma_tree,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
        final_epsilon=0.05,
        num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
        nd_update_freq=nd_update_freq_tree,
        robust=True,
        max_nd_size=nd_size_cap_tree,
        max_archive_size=archive_cap_tree,
        verbose=True,
        tag=file_end,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

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

    eval_df = evaluate_tree_policies_across_scenarios(
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
    # Persist the Q-table
    agent.save_q_table(f'{output_folder}/agent_{file_end}.pkl')

    return policies_df


def run_moro_lake(
        scoring, timesteps, ref_point, n_obj,
        num_weight_divisions, neighbourhood_size,
        output_folder, file_end, start_time=None,
):
    os.makedirs(output_folder, exist_ok=True)

    env = MoroLakeEnv(n_obj=n_obj, scenarios_path=lake_scenarios_path)

    agent = PQL(
        env=env, ref_point=ref_point, gamma=gamma_lake,
        initial_epsilon=1.0, epsilon_decay_steps=timesteps,
        final_epsilon=0.05, num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
        nd_update_freq=nd_update_freq_lake, robust=True,
        max_nd_size=nd_size_cap_lake,
        max_archive_size=archive_cap_lake,
        verbose=True,
        tag=file_end,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps, action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

    conv_df = pd.DataFrame(conv_log)
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    scenarios = env._scenarios
    n_scenarios = len(scenarios)

    def eval_env_factory(idx):
        s = scenarios[idx]
        return TwoLakeEnv(
            b1=float(s['b1']), q1=float(s['q1']),
            b2=float(s['b2']), q2=float(s['q2']),
            inflow_seed1=int(s['inflow_seed1']),
            inflow_seed2=int(s['inflow_seed2']),
            Pcrit1=float(s['Pcrit1']),
            Pcrit2=float(s['Pcrit2']),
            num_obj=n_obj,
            total_years=total_years,
            years_per_action=years_per_action,
        )

    # Multi-scenario robust re-evaluation — mirrors run_moro for tree
    eval_df = evaluate_lake_policies_across_scenarios(
        agent=agent,
        env_factory=eval_env_factory,
        n_scenarios=n_scenarios,
        n_obj=n_obj,
    )
    robust_pcs_df = compute_robustness(eval_df, n_obj)

    # Extract policy decisions on reference scenario (for policies_*.csv)
    policy_rows = []
    for pol_id, target_vec in enumerate(agent.archive):
        decisions = extract_lake_policy(agent, target_vec, eval_env_factory(0))
        row = {'policy_id': pol_id}
        for step, (u1, u2) in enumerate(decisions):
            row[f'u1_{step}'] = int(u1)
            row[f'u2_{step}'] = int(u2)
        policy_rows.append(row)

    policies_df = pd.DataFrame(policy_rows) if policy_rows else pd.DataFrame()
    policies_df.to_csv(f'{output_folder}/policies_{file_end}.csv', index=False)
    robust_pcs_df.to_csv(f'{output_folder}/pcs_{file_end}.csv', index=False)
    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)
    # Persist the Q-table
    agent.save_q_table(f'{output_folder}/agent_{file_end}.pkl')
    return policies_df

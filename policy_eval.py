import numpy as np
import pandas as pd
from count_non_dominated import is_nondominated


def extract_policy(agent, target_vec):

    env_shape = agent.env_shape
    depth = agent.env.unwrapped.tree_depth
    decisions = []
    target = np.array(target_vec)

    # If the agent was trained with decomposition scoring, the populated
    # bootstrap structure is `nd_decomp`, not `non_dominated`. Without this,
    # extract_policy reads zero-init phantom Q-sets and silently picks
    # action 0 at every level.
    decomp = getattr(agent, 'action_eval', None) == 'decomposition'

    level, pos = 0, 0
    for _ in range(depth):
        state = int(np.ravel_multi_index([level, pos], env_shape))
        best_action = None
        best_dist = np.inf
        next_target = target

        for a in range(agent.num_actions):
            if agent.counts[state][a] == 0:
                continue
            q_set = agent.get_q_set(state, a, decomp=decomp)
            for q_vec in q_set:
                dist = np.sum(np.abs(np.array(q_vec) - target))
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
                    next_target = np.array(q_vec) / agent.gamma \
                        if agent.gamma > 0 else np.array(q_vec)

        if best_action is None:
            best_action = 0  # fallback
        decisions.append(best_action)
        pos = pos * 2 + best_action
        level += 1
        target = next_target

    return decisions


def extract_lake_policy(agent, target_vec, env):

    target = np.array(target_vec)
    state, _ = env.reset()
    state_flat = int(np.ravel_multi_index(state, agent.env_shape))
    decisions = []
    terminated = False

    # See note in extract_policy.
    decomp = getattr(agent, 'action_eval', None) == 'decomposition'

    while not terminated:
        best_action = None
        best_dist = np.inf
        next_target = target

        for a in range(agent.num_actions):
            if agent.counts[state_flat][a] == 0:
                continue
            for q_vec in agent.get_q_set(state_flat, a, decomp=decomp):
                dist = np.sum(np.abs(np.array(q_vec) - target))
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
                    next_target = np.array(q_vec) / agent.gamma \
                        if agent.gamma > 0 else np.array(q_vec)

        if best_action is None:
            best_action = 0

        action_nd = np.unravel_index(best_action, env.action_space.nvec)
        decisions.append(action_nd)
        next_state, _, terminated, _, _ = env.step(np.array(action_nd))
        state_flat = int(np.ravel_multi_index(next_state, agent.env_shape))
        target = next_target

    return decisions


def evaluate_tree_policies_across_scenarios(agent, env_factory, n_scenarios):

    n_obj = agent.num_objectives
    rows = []

    if not agent.archive:
        return pd.DataFrame(columns=['policy_id', 'scenario_index'] +
                                    [f'o{i + 1}' for i in range(n_obj)])

    for pol_id, target_vec in enumerate(agent.archive):
        decisions = extract_policy(agent, target_vec)

        for scenario_idx in range(n_scenarios):
            env = env_factory(scenario_idx)
            env.reset()
            total_reward = np.zeros(n_obj)
            for action in decisions:
                _, reward, terminal, _, _ = env.step(action)
                total_reward += reward
                if terminal:
                    break

            row = {'policy_id': pol_id, 'scenario_index': scenario_idx}
            row.update({f'o{i + 1}': float(total_reward[i]) for i in range(n_obj)})
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_lake_policies_across_scenarios(agent, env_factory, n_scenarios,
                                            n_obj, ref_scenario_idx=0):

    rows = []
    if not agent.archive:
        return pd.DataFrame(columns=['policy_id', 'scenario_index'] +
                                    [f'o{i+1}' for i in range(n_obj)])

    for pol_id, target_vec in enumerate(agent.archive):
        # Extract action sequence once on the reference env
        ref_env = env_factory(ref_scenario_idx)
        decisions = extract_lake_policy(agent, target_vec, ref_env)

        for scenario_idx in range(n_scenarios):
            env = env_factory(scenario_idx)
            env.reset()
            total_reward = np.zeros(n_obj)
            for action_nd in decisions:
                _, reward, terminated, _, _ = env.step(np.array(action_nd))
                total_reward += reward
                if terminated:
                    break

            row = {'policy_id': pol_id, 'scenario_index': scenario_idx}
            row.update({f'o{i+1}': float(total_reward[i]) for i in range(n_obj)})
            rows.append(row)

    return pd.DataFrame(rows)


def compute_robustness(eval_df, n_obj, percentile=20):
    """
    Aggregate per-(policy, scenario) rewards into one robust score per
    policy via the given percentile across scenarios.
    """
    obj_cols = [f'o{i + 1}' for i in range(n_obj)]
    if eval_df.empty:
        return pd.DataFrame(columns=['policy_id'] + obj_cols)
    return (eval_df.groupby('policy_id')[obj_cols]
            .quantile(percentile / 100.0)
            .reset_index())


def evaluate_table_archive_robust(archive_path, depth, n_obj,
                                  env_factory, n_scenarios):

    df = pd.read_csv(archive_path)
    lever_cols = [c for c in df.columns if c.startswith('n')]
    if not lever_cols:
        raise ValueError(f"No lever columns (n0, n1, ...) found in {archive_path}.")

    n_internal = 2**depth - 1
    rows = []

    for pol_id, row in df.iterrows():
        table = [int(row[f'n{i}']) for i in range(n_internal)]

        for scenario_idx in range(n_scenarios):
            env = env_factory(scenario_idx)
            env.reset()
            total_reward = np.zeros(n_obj)
            for _ in range(depth):
                level, pos = env.current_state
                node_id = int(2**level - 1) + pos
                action = table[node_id]
                _, reward, terminal, _, _ = env.step(action)
                total_reward += reward
                if terminal:
                    break

            r = {'policy_id': pol_id, 'scenario_index': scenario_idx}
            r.update({f'o{i+1}': float(total_reward[i]) for i in range(n_obj)})
            rows.append(r)

    eval_df = pd.DataFrame(rows)
    robust_df = compute_robustness(eval_df, n_obj)

    if robust_df.empty:
        return robust_df

    robust_rewards = robust_df[[f'o{i+1}' for i in range(n_obj)]].values
    nd_mask = is_nondominated(robust_rewards)

    df_pruned = df.loc[robust_df.index[nd_mask]].copy()
    for i in range(n_obj):
        df_pruned[f'o{i+1}'] = robust_rewards[nd_mask, i]

    df_pruned = df_pruned.drop_duplicates(
        subset=[f'o{i+1}' for i in range(n_obj)]
    ).reset_index(drop=True)

    return df_pruned

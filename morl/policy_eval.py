import numpy as np
import pandas as pd


def extract_policy(agent, target_vec):
    """Trace back through the Q-table to recover the decision sequence
    for a given archive Q-vector.

    Args:
        agent: Trained PQL agent.
        target_vec: A Q-vector from agent.archive (the return to track).

    Returns:
        decisions: List of depth actions (0 or 1) forming the policy.
    """
    env_shape = agent.env_shape
    depth = agent.env.unwrapped.tree_depth
    decisions = []
    target = np.array(target_vec)

    level, pos = 0, 0
    for _ in range(depth):
        state = int(np.ravel_multi_index([level, pos], env_shape))
        best_action = None
        best_dist = np.inf
        next_target = target

        for a in range(agent.num_actions):
            if agent.counts[state][a] == 0:
                continue
            q_set = agent.get_q_set(state, a)
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


def evaluate_policies_across_scenarios(agent, env_factory, n_scenarios):
    n_obj = agent.num_objectives
    rows = []

    if not agent.archive:
        return pd.DataFrame(columns=['policy_id', 'scenario_index'] +
                            [f'o{i+1}' for i in range(n_obj)])

    for pol_id, target_vec in enumerate(agent.archive):
        decisions = extract_policy(agent, target_vec)

        for scenario_idx in range(n_scenarios):
            env = env_factory(scenario_idx)
            env.reset()
            reward = np.zeros(n_obj)
            for action in decisions:
                _, reward, terminal, _, _ = env.step(action)
                if terminal:
                    break

            row = {'policy_id': pol_id, 'scenario_index': scenario_idx}
            row.update({f'o{i + 1}': float(reward[i]) for i in range(n_obj)})
            rows.append(row)

    return pd.DataFrame(rows)


def compute_robustness(eval_df, n_obj):
    obj_cols = [f'o{i+1}' for i in range(n_obj)]

    if eval_df.empty:
        return pd.DataFrame(columns=['policy_id'] + obj_cols)

    records = []
    for pol_id, group in eval_df.groupby('policy_id'):
        row = {'policy_id': pol_id}
        for col in obj_cols:
            row[col] = np.percentile(group[col].values, 20)
        records.append(row)
    return pd.DataFrame(records)
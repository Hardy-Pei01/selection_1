import numpy as np
from fruit_tree import FruitTreeEnv

SEED = None


def fruit_tree_inter(depth, num_obj, csv_path, observe, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True)
    env.reset(SEED)

    total_reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal, _, _ = env.step(decisions[step])
        total_reward += reward
        if terminal:
            break

    return {f'o{i + 1}': -total_reward[i] for i in range(num_obj)}


def fruit_tree_table(depth, num_obj, csv_path, observe, **kwargs):
    n_internal = 2 ** depth - 1
    table = [int(kwargs[f'n{i}']) for i in range(n_internal)]

    env = FruitTreeEnv(depth=depth, reward_dim=num_obj,
                       csv_path=csv_path, observe=bool(observe))
    obs, _ = env.reset(SEED)

    total_reward = np.zeros(num_obj)
    for _ in range(depth):
        level, pos = obs
        node_id = int(2 ** level - 1) + pos
        action = table[node_id]
        obs, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        if terminal:
            break

    return {f'o{i + 1}': -total_reward[i] for i in range(num_obj)}


def fruit_tree_inter_robust(depth, num_obj, csv_path, observe,
                            scenario_index=0, slip_patterns_path=None, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path,
                       observe=True,
                       scenario_index=int(scenario_index),
                       slip_patterns_path=slip_patterns_path)
    env.reset()
    total_reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal, _, _ = env.step(decisions[step])
        total_reward += reward
        if terminal:
            break
    return {f'o{i + 1}': -total_reward[i] for i in range(num_obj)}


def fruit_tree_table_robust(depth, num_obj, csv_path, observe,
                            scenario_index=0, slip_patterns_path=None, **kwargs):
    n_internal = 2 ** depth - 1
    table = [int(kwargs[f'n{i}']) for i in range(n_internal)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path,
                       observe=bool(observe),
                       scenario_index=int(scenario_index),
                       slip_patterns_path=slip_patterns_path)
    obs, _ = env.reset()
    total_reward = np.zeros(num_obj)
    for _ in range(depth):
        level, pos = obs
        node_id = int(2 ** level - 1) + pos
        action = table[node_id]
        obs, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        if terminal:
            break
    return {f'o{i + 1}': -total_reward[i] for i in range(num_obj)}

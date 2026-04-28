import numpy as np
from fruit_tree import FruitTreeEnv

SEED = None


def fruit_tree_inter(depth, num_obj, csv_path, observe, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True)
    env.reset(SEED)

    reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal, _, _ = env.step(decisions[step])
        if terminal:
            break

    return {f'o{i + 1}': -reward[i] for i in range(num_obj)}


def fruit_tree_table(depth, num_obj, csv_path, observe, **kwargs):
    n_internal = 2 ** depth - 1
    table = [int(kwargs[f'n{i}']) for i in range(n_internal)]

    env = FruitTreeEnv(depth=depth, reward_dim=num_obj,
                       csv_path=csv_path, observe=bool(observe))
    env.reset(SEED)

    reward = np.zeros(num_obj)
    for _ in range(depth):
        level, pos = env.current_state
        node_id = int(2 ** level - 1) + pos
        action = table[node_id]
        _, reward, terminal, _, _ = env.step(action)
        if terminal:
            break

    return {f'o{i + 1}': -reward[i] for i in range(num_obj)}


def fruit_tree_inter_robust(depth, num_obj, csv_path, observe,
                            scenario_index=0, slip_patterns_path=None, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path,
                       observe=True,
                       scenario_index=int(scenario_index),
                       slip_patterns_path=slip_patterns_path)
    env.reset()
    reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal, _, _ = env.step(decisions[step])
        if terminal:
            break
    return {f'o{i + 1}': -reward[i] for i in range(num_obj)}


def fruit_tree_table_robust(depth, num_obj, csv_path, observe,
                            scenario_index=0, slip_patterns_path=None, **kwargs):
    n_internal = 2 ** depth - 1
    table = [int(kwargs[f'n{i}']) for i in range(n_internal)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path,
                       observe=bool(observe),
                       scenario_index=int(scenario_index),
                       slip_patterns_path=slip_patterns_path)
    env.reset()
    reward = np.zeros(num_obj)
    for _ in range(depth):
        level, pos = env.current_state
        node_id = int(2 ** level - 1) + pos
        action = table[node_id]
        _, reward, terminal, _, _ = env.step(action)
        if terminal:
            break
    return {f'o{i + 1}': -reward[i] for i in range(num_obj)}

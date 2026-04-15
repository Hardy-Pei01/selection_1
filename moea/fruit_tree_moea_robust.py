import numpy as np
from fruit_tree import FruitTreeEnv
SEED = None


def fruit_tree_inter_robust(depth, num_obj, csv_path, observe, slip_prob=0.0, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True, slip_prob=slip_prob)
    env.reset(SEED)

    reward = np.zeros(num_obj)
    for step in range(depth):
        action = decisions[step]
        _, reward, terminal = env.step(action)
        if terminal:
            break

    return {f'o{i + 1}': reward[i] for i in range(num_obj)}


def fruit_tree_table_robust(depth, num_obj, csv_path, observe, slip_prob=0.0, **kwargs):
    n_internal = 2 ** depth - 1
    table = [int(kwargs[f'n{i}']) for i in range(n_internal)]

    env = FruitTreeEnv(depth=depth, reward_dim=num_obj,
                       csv_path=csv_path, observe=bool(observe),
                       slip_prob=slip_prob)
    env.reset(SEED)

    reward = np.zeros(num_obj)
    for _ in range(depth):
        level, pos = env.current_state
        node_id = int(2 ** level - 1) + pos
        action = table[node_id]
        _, reward, terminal = env.step(action)
        if terminal:
            break

    return {f'o{i+1}': reward[i] for i in range(num_obj)}

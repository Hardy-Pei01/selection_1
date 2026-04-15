import numpy as np
from fruit_tree import FruitTreeEnv
SEED = 42


def get_action_linear(state, w0, w1, threshold, norm_scale):
    s = state / norm_scale
    return 1 if (w0 * s[0] + w1 * s[1]) > threshold else 0


def fruit_tree_inter(depth, num_obj, csv_path, observe, **kwargs):
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True)
    env.reset(SEED)

    reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal = env.step(decisions[step])
        if terminal:
            break

    return {f'o{i + 1}': reward[i] for i in range(num_obj)}


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
        _, reward, terminal = env.step(action)
        if terminal:
            break

    return {f'o{i+1}': reward[i] for i in range(num_obj)}

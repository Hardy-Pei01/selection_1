import numpy as np
from fruit_tree import FruitTreeEnv

SENSITIVITY = np.array([0.5, 0.1])


def get_weather_multiplier(w, sensitivity):
    return 1.0 + sensitivity * (2 * w - 1)


def get_action_linear(state, w0, w1, threshold, norm_scale):
    s = state / norm_scale
    return 1 if (w0 * s[0] + w1 * s[1]) > threshold else 0


def fruit_tree_inter_robust(depth, num_obj, csv_path, observe, w=0.5, **kwargs):
    sensitivity = np.repeat(SENSITIVITY, num_obj // 2)
    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True)
    env.reset()

    reward = np.zeros(num_obj)
    for step in range(depth):
        action = decisions[step]
        _, reward, terminal = env.step(action)
        if terminal:
            break

    final_reward = reward * get_weather_multiplier(w, sensitivity)
    return {f'o{i + 1}': final_reward[i] for i in range(num_obj)}


def fruit_tree_dps_robust(depth, num_obj, csv_path, observe, w=0.5, **kwargs):
    sensitivity = np.repeat(SENSITIVITY, num_obj // 2)

    w0 = kwargs['w0']
    w1 = kwargs['w1']
    threshold = kwargs['threshold']

    max_pos = 2 ** depth - 1
    norm_scale = np.array([depth, max_pos])

    env = FruitTreeEnv(
        depth=depth,
        reward_dim=num_obj,
        csv_path=csv_path,
        observe=bool(observe),
    )
    state = env.reset()

    reward = np.zeros(num_obj)
    for _ in range(depth):
        action = get_action_linear(state, w0, w1, threshold, norm_scale)
        state, reward, terminal = env.step(action)
        if terminal:
            break

    final_reward = reward * get_weather_multiplier(w, sensitivity)
    return {f'o{i + 1}': final_reward[i] for i in range(num_obj)}

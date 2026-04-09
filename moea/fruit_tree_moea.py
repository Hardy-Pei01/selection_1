import numpy as np
from fruit_tree import FruitTreeEnv


N_CENTERS = 4


def fruit_tree_inter(
        depth,
        num_obj,
        csv_path,
        observe,
        **kwargs
):

    decisions = [kwargs[f'l{i}'] for i in range(depth)]
    env = FruitTreeEnv(depth=depth, reward_dim=num_obj, csv_path=csv_path, observe=True)
    env.reset()

    reward = np.zeros(num_obj)
    for step in range(depth):
        _, reward, terminal = env.step(decisions[step])
        if terminal:
            break

    return {f'o{i+1}': reward[i] for i in range(len(reward))}


def get_action_rbf_improved(
    state: np.ndarray,
    centers: np.ndarray,     # (n_centers, 2)  — normalised coordinates
    radii: np.ndarray,       # (n_centers,)
    w_left: np.ndarray,      # (n_centers,)
    w_right: np.ndarray,     # (n_centers,)
    norm_scale: np.ndarray,  # (2,)  — [depth, max_position] for normalisation
) -> int:

    # Normalise state to [0, 1]
    s = state / norm_scale

    # Gaussian activations  φ_i = exp(−‖s − c_i‖² / r_i²)
    diffs = s - centers                                     # (n_centers, 2)
    sq_dists = np.sum(diffs ** 2, axis=1)                   # (n_centers,)
    safe_radii = np.maximum(radii, 1e-6)
    phi = np.exp(-sq_dists / (safe_radii ** 2))             # (n_centers,)

    score_left = np.dot(w_left, phi)
    score_right = np.dot(w_right, phi)

    return 0 if score_left >= score_right else 1


def fruit_tree_dps(depth, num_obj, csv_path, observe, **kwargs):

    n = N_CENTERS

    # Reconstruct policy parameters from the flat kwargs
    centers = np.array([[kwargs[f'c{i}_0'], kwargs[f'c{i}_1']]
                        for i in range(n)])
    radii = np.array([kwargs[f'rad{i}'] for i in range(n)])
    w_left = np.array([kwargs[f'wL{i}'] for i in range(n)])
    w_right = np.array([kwargs[f'wR{i}'] for i in range(n)])

    # Normalisation scale: row ∈ [0, depth], position ∈ [0, 2^depth − 1]
    max_pos = 2 ** depth - 1
    norm_scale = np.array([depth, max_pos])

    env = FruitTreeEnv(
        depth=depth,
        reward_dim=num_obj,
        csv_path=csv_path,
        observe=bool(observe),
    )
    state, _ = env.reset()

    reward = np.zeros(num_obj)
    for _ in range(depth):
        action = get_action_rbf_improved(
            state, centers, radii, w_left, w_right, norm_scale,
        )
        state, reward, terminal = env.step(action)
        if terminal:
            break

    return {f'o{i+1}': reward[i] for i in range(num_obj)}
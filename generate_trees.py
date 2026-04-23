import os

import numpy as np
import pandas as pd


def generate_leaf_rewards(depth, reward_dim, seed=None):
    rng = np.random.default_rng(seed)

    n_leaves = 2 ** depth
    rewards = rng.standard_normal((n_leaves, reward_dim))
    rewards = np.abs(rewards) / np.linalg.norm(rewards, 2, 1, True)
    rewards = rewards * -10.0
    rewards = np.round(rewards, 5)

    sort_keys = tuple(rewards[:, i] for i in range(reward_dim - 1, -1, -1))
    sort_order = np.lexsort(sort_keys)
    rewards = rewards[sort_order]

    columns = {f"r{i}": rewards[:, i] for i in range(reward_dim)}
    df = pd.DataFrame({"leaf_index": np.arange(n_leaves), **columns})
    return df


def main(depth, reward_dim, seed):
    df = generate_leaf_rewards(depth, reward_dim, seed)

    output_path = f"fruits/depth{depth}_dim{reward_dim}.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} leaves with {reward_dim} objectives "
          f"for depth {depth}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    depth = 10
    main(depth=depth, reward_dim=2, seed=1)
    # main(depth=depth, reward_dim=5, seed=1)
    main(depth=depth, reward_dim=6, seed=1)
    # main(depth=depth, reward_dim=11, seed=1)
    # main(depth=depth, reward_dim=14, seed=1)
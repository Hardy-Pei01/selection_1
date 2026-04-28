import os

import numpy as np
import pandas as pd

from count_non_dominated import is_nondominated

# ── Fixed seeds ───────────────────────────────────────────────────────────────
# Change only if you deliberately want a different benchmark instance.
TREE_SEED = 42  # Pareto-leaf hypersphere draw
DOMINATE_SEED = [7, 29, 98, 37, 41]  # dominated-leaf selection and offset sampling

# ── Benchmark hyperparameters ─────────────────────────────────────────────────
OFFSET_LOW = 0.5  # minimum per-objective offset subtracted from parent leaves
OFFSET_HIGH = 4.0  # maximum per-objective offset subtracted from parent leaves

BASE_DOMINATED_FRACTION = 0.25  # centre of the distribution
DOMINATED_FRACTION_NOISE = 0.1  # std — keeps fraction in ~[0.10, 0.40]
DOMINATED_FRACTION_MIN = 0.10  # hard floor
DOMINATED_FRACTION_MAX = 0.45  # hard ceiling


def generate_leaf_rewards(
        depth,
        reward_dim,
        offset_low=OFFSET_LOW,
        offset_high=OFFSET_HIGH,
        tree_seed=TREE_SEED,
        dominate_seed=DOMINATE_SEED[0],
):
    rng_tree = np.random.default_rng(tree_seed)
    rng_dom = np.random.default_rng(dominate_seed)

    n_leaves = 2 ** depth
    # Draw fraction from a separate RNG seeded by dominate_seed so it does
    # not shift the draws used for parent selection and offsets below.
    rng_frac = np.random.default_rng(dominate_seed)
    dominated_fraction = float(np.clip(
        BASE_DOMINATED_FRACTION + rng_frac.normal(0, DOMINATED_FRACTION_NOISE),
        DOMINATED_FRACTION_MIN,
        DOMINATED_FRACTION_MAX,
    ))
    n_dominated = int(n_leaves * dominated_fraction)
    n_good = n_leaves - n_dominated

    # ── Stage 1: Pareto-optimal leaves ───────────────────────────────────────
    # Sample directions uniformly on the positive unit hypersphere, then scale
    # to radius 10.  Every good leaf has norm = 10 and values in (0, 10].
    # No leaf dominates another under maximisation: if A[k] >= B[k] for all k
    # then norm(A) >= norm(B); equality forces A = B, so distinct sphere points
    # are mutually non-dominated.
    raw = rng_tree.standard_normal((n_good, reward_dim))
    raw = np.abs(raw) / np.linalg.norm(raw, axis=1, keepdims=True)
    good_rewards = raw * 10.0  # values in (0, 10], norm = 10

    # ── Stage 2: dominated leaves ─────────────────────────────────────────────
    # Each dominated leaf = (random Pareto parent) MINUS (positive per-obj offset).
    # Subtracting a strictly positive offset guarantees:
    #   dominated_leaf[j] < parent[j]  for every objective j
    # so the parent is larger on every objective → parent dominates the child
    # under maximisation.
    #
    # Using a per-objective (not scalar) offset creates asymmetric degradation:
    # a dominated leaf can still beat Pareto leaves on individual objectives,
    # preserving the non-trivial discrimination challenge.
    parent_idx = rng_dom.choice(n_good, size=n_dominated)
    parents = good_rewards[parent_idx]  # (n_dom, d)
    offsets = rng_dom.uniform(  # all > 0
        offset_low, offset_high,
        size=(n_dominated, reward_dim))
    dominated_rewards = np.clip(parents - offsets, 0.0, 10.0)

    # ── Stage 3: combine ──────────────────────────────────────────────────────
    all_rewards = np.concatenate([good_rewards, dominated_rewards], axis=0)
    intended_dominated = np.zeros(n_leaves, dtype=bool)
    intended_dominated[n_good:] = True

    all_rewards = np.round(all_rewards, 5)

    # ── Stage 4: verify ground-truth Pareto front ─────────────────────────────
    # A dominated leaf whose offset lands very close to zero on some objective
    # may end up non-dominated by accident (if its parent happened to be near
    # the boundary).  is_nondominated uses minimisation convention, so negate
    # rewards to convert to the maximisation problem used here.
    nd_mask = is_nondominated(all_rewards)

    # ── Sort by objectives descending (best first for readability) ────────────
    sort_keys = tuple(-all_rewards[:, i] for i in range(reward_dim - 1, -1, -1))
    sort_order = np.lexsort(sort_keys)
    all_rewards = all_rewards[sort_order]
    nd_mask = nd_mask[sort_order]
    intended_dominated = intended_dominated[sort_order]

    columns = {f"r{i}": all_rewards[:, i] for i in range(reward_dim)}
    df = pd.DataFrame({
        "leaf_index": np.arange(n_leaves),
        **columns,
        "is_dominated": intended_dominated.astype(int),
        "ground_truth_pareto": nd_mask.astype(int),
    })
    return df


def main(
        depth,
        reward_dim,
        tree_seed=TREE_SEED,
        dominate_seed=DOMINATE_SEED[0],
):
    df = generate_leaf_rewards(
        depth=depth,
        reward_dim=reward_dim,
        tree_seed=tree_seed,
        dominate_seed=dominate_seed,
    )

    output_path = f"fruits/depth{depth}_dim{reward_dim}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    n_total = len(df)
    n_nd = int(df["ground_truth_pareto"].sum())
    n_intended = int(df["is_dominated"].sum())

    print(f"depth={depth}  reward_dim={reward_dim}")
    print(f"  Total leaves          : {n_total}")
    print(f"  Intended dominated    : {n_intended}  ({n_intended / n_total:.1%})")
    print(f"  Ground-truth Pareto   : {n_nd}  ({n_nd / n_total:.1%})")
    print(f"  Ground-truth dominated: {n_total - n_nd}  ({(n_total - n_nd) / n_total:.1%})")
    if abs(n_intended / n_total - (n_total - n_nd) / n_total) > 0.05:
        print(f"  WARNING: actual dominated fraction ({(n_total - n_nd) / n_total:.1%}) "
              f"differs from target ({n_intended / n_total:.1%}) by more than 5 pp.")
    print(f"  Seeds: tree={tree_seed}, dominate={dominate_seed}")
    print(f"  Saved → {output_path}")
    print()


if __name__ == "__main__":
    depth = 11
    main(depth=depth, reward_dim=2, tree_seed=TREE_SEED, dominate_seed=DOMINATE_SEED[0])
    # main(depth=depth, reward_dim=4, tree_seed=TREE_SEED, dominate_seed=DOMINATE_SEED[2])
    main(depth=depth, reward_dim=6, tree_seed=TREE_SEED, dominate_seed=DOMINATE_SEED[1])
    # main(depth=depth, reward_dim=8, tree_seed=TREE_SEED, dominate_seed=DOMINATE_SEED[3])
    # main(depth=depth, reward_dim=10, tree_seed=TREE_SEED, dominate_seed=DOMINATE_SEED[4])
"""
scenario_selection.py
─────────────────────
Eker–Kwakkel (2018) / Bartholomew–Kwakkel (2020) reference-scenario selection,
adapted for the two-lake problem.

Single output file, valid for both 2-obj and 6-obj multi-scenario MORDM runs.
Rationale: the lakes are independent, the env always uses all 6 deep-uncertainty
values regardless of num_obj, and selection in 6-obj outcome space implies
policy-relevance for lake 1 alone (the 2-obj sub-problem) since both lakes
bad ⇒ lake 1 bad. Diversity in 6-D projects acceptably onto the 2-D subspace
when the dimensions are independent.

Procedure:
  1. Generate N LHS samples over the 6 deep uncertainties.
  2. Re-evaluate every solution from a MORDM archive, env always at num_obj=6.
  3. Aggregate per scenario: median outcome vector across the archive.
  4. Filter to policy-relevant scenarios — all 6 outcomes in undesired half
     (env returns all rewards in maximize-direction, so undesired = below median).
  5. Pick K=4 maximally diverse scenarios via Carlsen et al. (2016) metric
     (w=0.5) over normalised outcome distances. Exhaustive search if
     (M choose K) ≤ 1e6, otherwise greedy farthest-point + hill-climb.

Usage:
  python scenario_selection.py --archive ARCHIVE.csv --out selected.csv
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from two_lake import TwoLakeEnv

# ── Config ────────────────────────────────────────────────────────────────────

UNCERTAINTY_BOUNDS = {
    "b1":           (0.10, 0.45),
    "q1":           (2.0,  4.5),
    "b2":           (0.10, 0.45),
    "q2":           (2.0,  4.5),
    "inflow_seed1": (0,    10000),
    "inflow_seed2": (0,    10000),
}
INT_DIMS = {"inflow_seed1", "inflow_seed2"}

# All env rewards are maximize-direction.
OUTCOME_MAXIMIZE = True


# ── LHS sampling over the deep-uncertainty box ────────────────────────────────

def lhs(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube sample on [0, 1]^n_dims, shape (n_samples, n_dims)."""
    cuts = np.linspace(0, 1, n_samples + 1)
    u = rng.uniform(cuts[:-1], cuts[1:], size=(n_dims, n_samples)).T
    for j in range(n_dims):
        rng.shuffle(u[:, j])
    return u


def generate_scenarios(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = lhs(n, len(UNCERTAINTY_BOUNDS), rng)
    cols = {}
    for j, (name, (lo, hi)) in enumerate(UNCERTAINTY_BOUNDS.items()):
        if name in INT_DIMS:
            # +1 because hi is inclusive for integers
            cols[name] = (lo + u[:, j] * (hi - lo + 1)).astype(int)
        else:
            cols[name] = lo + u[:, j] * (hi - lo)
    return pd.DataFrame(cols)


# ── Archive loading & evaluation ──────────────────────────────────────────────

def load_intertemporal_archive(path: str) -> list[list[tuple[int, int]]]:
    """Each row → list of (u1, u2) actions. Skips outcome/metadata columns."""
    df = pd.read_csv(path)
    n_steps = sum(c.startswith("u1_") for c in df.columns)
    if n_steps == 0:
        raise ValueError(f"No 'u1_*' columns in {path}; is this an intertemporal archive?")
    archive = []
    for _, row in df.iterrows():
        archive.append([(int(row[f"u1_{i}"]), int(row[f"u2_{i}"])) for i in range(n_steps)])
    return archive


def _run_episode(env, actions, num_obj: int) -> np.ndarray:
    env.reset()
    total = np.zeros(num_obj, dtype=np.float64)
    for u1, u2 in actions:
        _, r, _, _, _ = env.step(np.array([u1, u2]))
        total += r
    return total


def reevaluate_all(archive, scenarios_df, num_obj: int) -> np.ndarray:
    """→ shape (n_scenarios, n_archive, num_obj). Builds env once per scenario."""
    out = np.empty((len(scenarios_df), len(archive), num_obj))
    for s_idx in range(len(scenarios_df)):
        s = scenarios_df.iloc[s_idx]
        env = TwoLakeEnv(
            b1=float(s["b1"]), q1=float(s["q1"]),
            b2=float(s["b2"]), q2=float(s["q2"]),
            inflow_seed1=int(s["inflow_seed1"]),
            inflow_seed2=int(s["inflow_seed2"]),
            num_obj=num_obj,
        )
        for p_idx, actions in enumerate(archive):
            out[s_idx, p_idx] = _run_episode(env, actions, num_obj)
    return out


# ── Selection ────────────────────────────────────────────────────────────────

def filter_policy_relevant(scenario_outcomes: np.ndarray) -> np.ndarray:
    """Mask of scenarios in the undesired half for ALL outcomes simultaneously.

    For maximise objectives, undesired = below median. (env rewards are all max-dir.)
    """
    medians = np.median(scenario_outcomes, axis=0)
    return np.all(scenario_outcomes <= medians, axis=1)


def normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(axis=0), arr.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return (arr - mn) / rng


def carlsen_diversity(subset: np.ndarray, w: float = 0.5) -> float:
    """D = (1-w)·min(pairwise) + w·mean(pairwise). Higher = more diverse."""
    n = len(subset)
    if n < 2:
        return 0.0
    dists = [np.linalg.norm(subset[i] - subset[j]) for i, j in combinations(range(n), 2)]
    dists = np.array(dists)
    return (1 - w) * dists.min() + w * dists.mean()


def select_diverse(outcomes_norm: np.ndarray, k: int = 4, w: float = 0.5,
                   exhaustive_cap: int = 1_000_000):
    """Pick k indices maximising Carlsen diversity.

    Uses exhaustive search if (n choose k) ≤ exhaustive_cap, otherwise falls
    back to greedy farthest-point seeding refined by a few hill-climbing swaps.
    Returns (indices, score, mode) where mode ∈ {'exhaustive', 'greedy'}.
    """
    from math import comb
    n = len(outcomes_norm)
    if n < k:
        raise ValueError(f"Need ≥{k} candidates; got {n}.")

    if comb(n, k) <= exhaustive_cap:
        best, best_idx = -np.inf, None
        for combo in combinations(range(n), k):
            d = carlsen_diversity(outcomes_norm[list(combo)], w=w)
            if d > best:
                best, best_idx = d, combo
        return list(best_idx), best, "exhaustive"

    # ---- Greedy farthest-point seeding ----
    # Seed with the two most distant points; then add the point whose minimum
    # distance to the current set is largest (max-min); repeat until k.
    rng = np.random.default_rng(0)
    pairwise = np.linalg.norm(
        outcomes_norm[:, None, :] - outcomes_norm[None, :, :], axis=-1
    )
    i, j = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < k:
        rest = [x for x in range(n) if x not in chosen]
        min_d = pairwise[rest][:, chosen].min(axis=1)
        chosen.append(int(rest[int(np.argmax(min_d))]))

    # ---- Hill-climbing refinement: try swapping each chosen with each non-chosen ----
    best = carlsen_diversity(outcomes_norm[chosen], w=w)
    improved = True
    while improved:
        improved = False
        for slot in range(k):
            for cand in range(n):
                if cand in chosen:
                    continue
                trial = chosen.copy()
                trial[slot] = cand
                d = carlsen_diversity(outcomes_norm[trial], w=w)
                if d > best + 1e-12:
                    chosen, best = trial, d
                    improved = True
    return chosen, best, "greedy"


# ── End-to-end ────────────────────────────────────────────────────────────────

def run(archive_path: str, n_scenarios: int = 500,
        k: int = 4, w: float = 0.5, seed: int = 42,
        out_path: str = "selected_scenarios.csv"):
    """Always evaluates at num_obj=6. The output file is valid for both 2-obj
    and 6-obj multi-scenario MORDM runs (lakes independent → 6-obj
    policy-relevance implies 2-obj policy-relevance)."""
    n_obj = 6

    print(f"Loading archive: {archive_path}")
    archive = load_intertemporal_archive(archive_path)
    print(f"  {len(archive)} solutions, {len(archive[0])} action steps each")

    print(f"Generating {n_scenarios} LHS scenarios...")
    scenarios = generate_scenarios(n_scenarios, seed=seed)

    print(f"Re-evaluating {len(archive)} × {n_scenarios} = "
          f"{len(archive) * n_scenarios} episodes...")
    outcomes_3d = reevaluate_all(archive, scenarios, num_obj=n_obj)

    print("Aggregating per scenario (median across archive)...")
    per_sc = np.median(outcomes_3d, axis=1)  # (n_scenarios, n_obj)

    mask = filter_policy_relevant(per_sc)
    M = int(mask.sum())
    print(f"Policy-relevant subset: M = {M} of {n_scenarios} "
          f"(expected ≈ {n_scenarios / 2**n_obj:.0f} for {n_obj} objs)")

    if M < k:
        raise RuntimeError(
            f"M={M} < K={k}. Increase --n-scenarios, or relax the filter "
            f"(e.g. require only n_obj-1 outcomes in undesired half)."
        )

    relevant_idx = np.where(mask)[0]
    relevant_outcomes = per_sc[mask]

    from math import comb
    n_combos = comb(M, k)
    print(f"({M} choose {k}) = {n_combos:,} subsets — "
          f"{'exhaustive' if n_combos <= 1_000_000 else 'greedy + hill-climb'}")
    selected_local, score, mode = select_diverse(normalise(relevant_outcomes), k=k, w=w)
    selected_global = relevant_idx[selected_local]

    print(f"  Carlsen diversity score: {score:.4f}  (mode: {mode})")
    selected_df = scenarios.iloc[selected_global].copy().reset_index(drop=True)
    selected_df.insert(0, "scenario_id", range(1, k + 1))

    print("\nSelected scenarios:")
    print(selected_df.to_string(index=False))

    selected_df.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return selected_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True, help="Path to MORDM archive CSV (intertemporal)")
    p.add_argument("--n-scenarios", type=int, default=500)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--w", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="selected_scenarios.csv")
    args = p.parse_args()
    run(args.archive, args.n_scenarios, args.k, args.w, args.seed, args.out)
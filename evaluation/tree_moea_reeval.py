"""Re-evaluate robust MOEA tree policies (multi and moro) on a held-out
evaluation scenario set.

Re-evaluation runs per seed folder:
  1. Collect archive files in the seed folder. multi has 5 ref-scenario
     files (ref_num 0..4); moro has 1. Skip *_pruned.csv and *_evaluated.csv.
  2. Merge rows from all files; for multi, dedup by decision-variable
     fingerprint so a policy trained on different reference scenarios is
     evaluated only once.
  3. Re-evaluate every distinct policy under all evaluation scenarios:
     mean of each objective across scenarios.
  4. Drop dominated rows (Pareto filter, minimization form).
  5. Write `archives_{stem}_{nfe}_evaluated.csv` into the same seed folder.

Output columns: policy_id, the decision columns, o1_mean..on_mean. The
reference_scenario field from multi archives is dropped (irrelevant after
merging). Objectives are in MIN-form (negated rewards) to match the
training-archive sign convention.

MOEA policies are read directly:
  - intertemporal: levers l0..l{depth-1} = fixed action sequence.
  - table:         levers n0..n{2^depth-2} = state-indexed action table.

Usage:
    python evaluate_tree_moea_robust.py
"""
import os
import re
import time

import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None


# ── Configuration ─────────────────────────────────────────────────────────────
MOEA_BASE = '../tree_data/robust/moea'
EVAL_PATTERNS_PATH = '../trees/slip_patterns_depth9_eval.npy'
CSV_PATH_DIM = {
    2: '../trees/depth9_dim2.csv',
    6: '../trees/depth9_dim6.csv',
}
TREE_DEPTH = 9

# Cell folder → (policy_kind, algo, method, n_obj, obs). Only multi/moro robust.
FOLDER_RE = re.compile(
    r'^(intertemporal|table)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)_'
    r'(observable|non_observable)$'
)
# Archive files: archives_{stem}_{nfe}[_{ref_num}].csv
ARCHIVE_RE = re.compile(r'^archives_(.+?)_(\d+)(?:_(\d+))?\.csv$')


# ── Policy rollout ────────────────────────────────────────────────────────────
def rollout_intertemporal(env, decisions):
    """Apply a fixed action sequence under the env's current slip pattern.
    Returns the per-objective sum, NEGATED (minimization convention).
    """
    env.reset()
    total = np.zeros(env.reward_dim)
    for action in decisions:
        _, reward, terminal, _, _ = env.step(int(action))
        total += reward
        if terminal:
            break
    return -total


def rollout_table(env, table, depth):
    """Apply a state-indexed table policy under the env's current slip
    pattern: at each step look up the action by node_id = 2^level - 1 + pos.
    Returns the per-objective sum, NEGATED (minimization convention).
    """
    obs, _ = env.reset()
    total = np.zeros(env.reward_dim)
    for _ in range(depth):
        level, pos = obs
        node_id = int(2 ** level - 1) + pos
        action = int(table[node_id])
        obs, reward, terminal, _, _ = env.step(action)
        total += reward
        if terminal:
            break
    return -total


# ── Pareto filter (minimization form) ─────────────────────────────────────────
def _filter_non_dominated(values):
    """Boolean mask of non-dominated rows. Inputs are minimization
    objectives (smaller is better — matches the negative-reward convention).
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)
    if _moocore_is_nd is not None:
        return _moocore_is_nd(arr)
    # Fallback O(n^2) — fine for archive-sized inputs.
    n = arr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                keep[i] = False
                break
    return keep


# ── Per-seed evaluation ───────────────────────────────────────────────────────
def evaluate_seed_folder(seed_dir, policy_kind, n_obj, depth, csv_path,
                         eval_patterns):
    """Evaluate all policies in one seed folder.

    Returns (out_df, n_before_dedup, n_after_dedup, n_dominated_dropped,
             archive_stem, archive_nfe), or None if no archive files.
    """
    # Decision columns for this policy kind.
    if policy_kind == 'intertemporal':
        dec_cols = [f'l{i}' for i in range(depth)]
    elif policy_kind == 'table':
        dec_cols = [f'n{i}' for i in range(2 ** depth - 1)]
    else:
        raise ValueError(f'unknown policy_kind: {policy_kind!r}')

    # Collect archive files (skip our own outputs and any pruned files).
    archive_files, stem, nfe = [], None, None
    for fname in sorted(os.listdir(seed_dir)):
        if fname.endswith('_evaluated.csv') or fname.endswith('_pruned.csv'):
            continue
        am = ARCHIVE_RE.match(fname)
        if not am:
            continue
        stem, nfe, _ref_num = am.groups()
        archive_files.append(os.path.join(seed_dir, fname))
    if not archive_files:
        return None

    # Merge decision columns from all files; dedup identical policies.
    parts = []
    for fpath in archive_files:
        df = pd.read_csv(fpath)
        missing = [c for c in dec_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f'{fpath} missing decision columns: {missing[:5]}...')
        parts.append(df[dec_cols])
    merged = pd.concat(parts, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=dec_cols, ignore_index=True)
    n_after = len(merged)

    # One env per seed folder; slip pattern swapped in per scenario.
    env = FruitTreeEnv(
        depth=depth, reward_dim=n_obj, csv_path=csv_path,
        observe=True, scenario_index=None, slip_patterns_path=None,
    )

    n_scen = len(eval_patterns)
    means = np.zeros((len(merged), n_obj))
    for i, row in merged.iterrows():
        decisions = row.values
        scenario_returns = np.zeros((n_scen, n_obj))
        for s in range(n_scen):
            env._slip_pattern = eval_patterns[s]
            if policy_kind == 'intertemporal':
                scenario_returns[s] = rollout_intertemporal(env, decisions)
            else:
                scenario_returns[s] = rollout_table(env, decisions, depth)
        means[i] = scenario_returns.mean(axis=0)

    # Pareto filter on the mean objectives (already minimization form).
    nd_mask = _filter_non_dominated(means)
    n_dom = int((~nd_mask).sum())

    out = merged[nd_mask].reset_index(drop=True).copy()
    means_nd = means[nd_mask]
    for j in range(n_obj):
        out[f'o{j + 1}_mean'] = means_nd[:, j]
    out.insert(0, 'policy_id', np.arange(len(out)))

    return out, n_before, n_after, n_dom, stem, nfe


# ── Walker ────────────────────────────────────────────────────────────────────
def walk_and_evaluate(base, csv_path_dim, eval_patterns_path):
    if not os.path.isdir(base):
        raise SystemExit(f'MOEA base not found: {base}')
    if not os.path.exists(eval_patterns_path):
        raise SystemExit(f'eval patterns not found: {eval_patterns_path}')

    eval_patterns = np.load(eval_patterns_path)
    print(f'Loaded {len(eval_patterns)} evaluation scenarios '
          f'({eval_patterns.shape[1]} nodes each)\n')

    # Discover robust cells.
    cells = []
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        policy_kind, algo, method, n_obj, obs = m.groups()
        cells.append((d, policy_kind, algo, method, int(n_obj), obs))
    if not cells:
        raise SystemExit(f'No multi/moro cells found under {base}')
    print(f'Found {len(cells)} robust MOEA cell(s)\n')

    t0 = time.time()
    n_seeds_done = 0
    for cell_dir, policy_kind, algo, method, n_obj, obs in cells:
        print(f'{cell_dir}  [{policy_kind}, {algo}, {method}, {n_obj}-obj]')
        cell_path = os.path.join(base, cell_dir)
        csv_path = csv_path_dim[n_obj]

        seed_folders = sorted(
            sd for sd in os.listdir(cell_path)
            if os.path.isdir(os.path.join(cell_path, sd))
        )
        for sd in seed_folders:
            seed_dir = os.path.join(cell_path, sd)
            result = evaluate_seed_folder(
                seed_dir, policy_kind, n_obj, TREE_DEPTH,
                csv_path, eval_patterns)
            if result is None:
                print(f'    {sd}: no archive files — skipped')
                continue
            out, n_before, n_after, n_dom, stem, nfe = result
            out_name = f'archives_{stem}_{nfe}_evaluated.csv'
            out.to_csv(os.path.join(seed_dir, out_name), index=False)
            print(f'    {sd}: {n_before} rows → {n_after} unique '
                  f'({n_before - n_after} dup) → {len(out)} non-dominated '
                  f'({n_dom} dom) → {out_name}')
            n_seeds_done += 1
        print()

    print(f'Done. {n_seeds_done} seed folder(s) written in '
          f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    walk_and_evaluate(
        base=MOEA_BASE,
        csv_path_dim=CSV_PATH_DIM,
        eval_patterns_path=EVAL_PATTERNS_PATH,
    )

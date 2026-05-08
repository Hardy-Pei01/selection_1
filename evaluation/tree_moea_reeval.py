"""Re-evaluate MOEA tree policies (multi and moro) under a held-out evaluation
scenario set.

For each (policy, algo, method, n_obj, obs) cell:
  1. Collect all archive files in the folder (multi has 5 ref-scenario files;
     moro has 1; ignore *_pruned.csv).
  2. Merge rows from all files. For multi, dedup by decision-variable
     fingerprint so identical policies trained on different reference
     scenarios are evaluated only once.
  3. Re-evaluate every distinct policy under all evaluation scenarios:
     mean of each objective across scenarios.
  4. Drop dominated rows (Pareto filter in minimization form).
  5. Save one file per cell: `archives_{stem}_{nfe}_evaluated.csv`.

Output columns: a `policy_id` index, the decision columns, and
`o1_mean..on_mean`. The `reference_scenario` field from multi archives is
not preserved (it's irrelevant after merging).

Usage:
    python evaluate_moea_robust.py
"""
import os
import re
import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None


# ── Folder/file pattern parsing ───────────────────────────────────────────────
FOLDER_RE = re.compile(
    r'^(intertemporal|table)_(NSGAII|IBEA|MOEAD)_(multi|moro)_(\d+)_(observable|non_observable)$'
)

# Archive files: archives_{stem}_{nfe}[_{ref_num}].csv
ARCHIVE_RE = re.compile(
    r'^archives_(.+?)_(\d+)(?:_(\d+))?\.csv$'
)


# ── Policy rollout ────────────────────────────────────────────────────────────
def rollout_intertemporal(env, decisions):
    """Apply the action sequence under the env's current slip pattern.
    Returns the per-objective sum, negated (to match archive minimization
    convention).
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
    """Apply the table policy: at each step, look up the action by
    node_id = 2^level - 1 + pos. Returns the per-objective sum, negated.
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
    """Return a boolean mask of non-dominated rows. Inputs are minimization
    objectives (smaller is better — matches the negative-reward convention).
    """
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)

    if _moocore_is_nd is not None:
        return _moocore_is_nd(arr)

    # Fallback O(n^2) implementation if moocore is missing.
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


# ── Per-cell evaluation ───────────────────────────────────────────────────────
def evaluate_cell(folder_path, policy_kind, n_obj, depth, csv_path,
                  eval_patterns, archive_files):
    """Evaluate all policies in one cell. Returns (out_df, n_before, n_after_dedup,
    n_dominated_dropped).

    archive_files: list of (file_path, ref_num) tuples. For moro, one entry
    with ref_num=None. For multi, five entries with ref_num=0..4.
    """
    if policy_kind == 'intertemporal':
        dec_cols = [f'l{i}' for i in range(depth)]
    elif policy_kind == 'table':
        n_internal = 2 ** depth - 1
        dec_cols = [f'n{i}' for i in range(n_internal)]
    else:
        raise ValueError(f"unknown policy_kind: {policy_kind!r}")

    # Read all archives, concatenate decision columns, drop duplicates
    parts = []
    for fpath, _ref_num in archive_files:
        df = pd.read_csv(fpath)
        missing = [c for c in dec_cols if c not in df.columns]
        if missing:
            raise ValueError(f'{fpath} missing decision columns: {missing[:5]}...')
        parts.append(df[dec_cols])

    merged = pd.concat(parts, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=dec_cols, ignore_index=True)
    n_after = len(merged)

    # Construct one env, override slip_pattern per scenario
    env = FruitTreeEnv(
        depth=depth, reward_dim=n_obj,
        csv_path=csv_path,
        observe=True,
        scenario_index=None,
        slip_patterns_path=None,
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

    # Pareto filter on the mean objectives (minimization form)
    nd_mask = _filter_non_dominated(means)
    n_dominated_dropped = int((~nd_mask).sum())

    out = merged[nd_mask].reset_index(drop=True).copy()
    means_nd = means[nd_mask]
    for j in range(n_obj):
        out[f'o{j + 1}_mean'] = means_nd[:, j]

    out.insert(0, 'policy_id', np.arange(len(out)))

    return out, n_before, n_after, n_dominated_dropped


# ── Walker ────────────────────────────────────────────────────────────────────
def walk_and_evaluate(base, csv_path_dim, eval_patterns_path):
    eval_patterns = np.load(eval_patterns_path)
    print(f'Loaded {len(eval_patterns)} evaluation scenarios from {eval_patterns_path}')

    n_cells = 0
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        policy_kind, algo, method, n_obj, obs = m.groups()
        n_obj = int(n_obj)
        folder_path = os.path.join(base, d)
        csv_path = csv_path_dim[n_obj]

        archive_files = []
        archive_stem = None
        archive_nfe = None
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('_evaluated.csv'):
                continue
            am = ARCHIVE_RE.match(fname)
            if not am:
                continue
            if fname.endswith('_pruned.csv'):
                continue
            stem, nfe, ref_num = am.groups()
            ref_num = int(ref_num) if ref_num is not None else None
            archive_files.append((os.path.join(folder_path, fname), ref_num))
            archive_stem = stem
            archive_nfe = nfe

        if not archive_files:
            continue

        evaluated, n_before, n_after, n_dom = evaluate_cell(
            folder_path=folder_path,
            policy_kind=policy_kind,
            n_obj=n_obj, depth=9,
            csv_path=csv_path,
            eval_patterns=eval_patterns,
            archive_files=archive_files,
        )

        out_name = f'archives_{archive_stem}_{archive_nfe}_evaluated.csv'
        out_path = os.path.join(folder_path, out_name)
        evaluated.to_csv(out_path, index=False)

        print(f'  {d}: {len(archive_files)} file(s), '
              f'{n_before} → {n_after} after dedup '
              f'({n_before - n_after} dup), '
              f'{n_after} → {len(evaluated)} after ND filter '
              f'({n_dom} dom) → {out_name}')
        n_cells += 1

    print(f'\nDone. {n_cells} cells written.')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE = '../tree_data/moea_robust'
    EVAL_PATTERNS_PATH = '../trees/slip_patterns_depth9_eval.npy'
    CSV_PATH_DIM = {
        2: '../trees/depth9_dim2.csv',
        6: '../trees/depth9_dim6.csv',
    }

    walk_and_evaluate(
        base=BASE,
        csv_path_dim=CSV_PATH_DIM,
        eval_patterns_path=EVAL_PATTERNS_PATH,
    )
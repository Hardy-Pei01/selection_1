"""Re-evaluate robust MOEA policies for the CONSTRAINED two-lake problem
(multi and moro) on a held-out evaluation scenario set.

Ported from the unconstrained-lake re-evaluation program. The only
problem-specific change is the import: `TwoLakeEnv` comes from
`constrained_two_lake` (the constrained env folds a MAX_P_THRESHOLD
violation penalty into every reward axis). Everything else is unchanged —
the constrained env has the same observation/action spaces, the same
constructor signature, and the same `n_gym_steps` as the unconstrained one;
the constraint only changes the reward values, which re-evaluation measures.

MOEA policies are read directly from the archive CSVs:
  - intertemporal: per-step emission columns u1_0..u1_{T-1}, u2_0..u2_{T-1}.
  - dps:           RBF controller parameters c1_1..w1_2 — emission computed
                   from the lake state each step via the cubic RBF rule.

Per cell:
  1. Collect archive files. multi has 5 ref-scenario files; moro has 1.
     Skip *_evaluated.csv and *_pruned.csv.
  2. Merge rows, dedup by decision columns (auto-detected: everything that
     isn't an objective column `o*`/`p*_o*` or the `reference_scenario`
     meta-column).
  3. Re-evaluate every distinct policy across all eval scenarios; mean of
     each objective.
  4. Drop dominated rows (Pareto filter, minimization form).
  5. Write `archives_{stem}_{nfe}_evaluated.csv` into the cell folder.

Output objective columns o1_mean..on_mean are in MIN-form (positive cost =
worse), matching the convention shared with the MORL re-evaluation output.

Usage:
    python evaluate_constrained_lake_moea_robust.py \
        --base ../constrained_lake_data/robust/moea \
        --eval-scenarios ../lakes/lake_scenarios_eval.npy
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from constrained_two_lake import ConstrainedTwoLakeEnv as TwoLakeEnv

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None

FOLDER_RE = re.compile(
    r'^(intertemporal|dps)'
    r'_(NSGAII|IBEA|MOEAD)'
    r'_(multi|moro)'
    r'_(\d+)(?:_seed(\d+))?$'
)

ARCHIVE_RE = re.compile(r'^archives_(.+?)_(\d+)(?:_(\d+))?\.csv$')

# Objective columns: o1, o2, ... or p20_o1, p20_o2, ... (the robust
# percentile-aggregated objective columns the MOEA writes).
_OBJ_COL_RE = re.compile(r'^(o\d+|p\d+_o\d+)$')
_META_COLS = {'reference_scenario'}


def detect_dec_cols(df: pd.DataFrame) -> List[str]:
    """Decision columns = everything that isn't an objective column, a known
    meta-column, or an unnamed index column.
    """
    return [
        c for c in df.columns
        if c not in _META_COLS
        and not _OBJ_COL_RE.match(c)
        and not c.startswith('Unnamed')
    ]


def build_env(scenario: np.void, n_obj: int) -> TwoLakeEnv:
    """Build a constrained TwoLakeEnv for one evaluation scenario."""
    return TwoLakeEnv(
        b1=float(scenario['b1']),
        q1=float(scenario['q1']),
        b2=float(scenario['b2']),
        q2=float(scenario['q2']),
        inflow_seed1=int(scenario['inflow_seed1']),
        inflow_seed2=int(scenario['inflow_seed2']),
        Pcrit1=float(scenario['Pcrit1']),
        Pcrit2=float(scenario['Pcrit2']),
        num_obj=n_obj,
    )


# ── Rollout functions ─────────────────────────────────────────────────────────
def rollout_intertemporal(env: TwoLakeEnv, row: pd.Series) -> np.ndarray:
    """Apply the fixed per-step emission sequence. Returns the per-objective
    summed cost (negated env reward: positive cost = worse).
    """
    env.reset()
    n_steps = env.n_gym_steps
    total = np.zeros(env.num_obj, dtype=np.float64)
    for i in range(n_steps):
        u1 = int(row[f'u1_{i}'])
        u2 = int(row[f'u2_{i}'])
        _, rewards, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total -= rewards  # negate: positive cost = worse
    return total


def _get_emission(xt: float,
                  c1: float, c2: float,
                  r1: float, r2: float,
                  w1: float) -> int:
    """Cubic RBF controller: map lake state xt to a discrete emission bin."""
    rule = (w1 * (abs(xt - c1) / r1) ** 3
            + (1 - w1) * (abs(xt - c2) / r2) ** 3)
    u = float(np.clip(rule, 0.0, 0.10))
    return int(round(u / 0.02))


def rollout_dps(env: TwoLakeEnv, row: pd.Series) -> np.ndarray:
    """Apply the DPS/RBF closed-loop controller. Emission each step is a
    function of the current lake state. Returns per-objective summed cost.
    """
    env.reset()
    c1_1, c2_1 = float(row['c1_1']), float(row['c2_1'])
    r1_1, r2_1 = float(row['r1_1']), float(row['r2_1'])
    w1_1 = float(row['w1_1'])
    c1_2, c2_2 = float(row['c1_2']), float(row['c2_2'])
    r1_2, r2_2 = float(row['r1_2']), float(row['r2_2'])
    w1_2 = float(row['w1_2'])

    total = np.zeros(env.num_obj, dtype=np.float64)
    for _ in range(env.n_gym_steps):
        X1, X2 = env.X1, env.X2
        u1 = _get_emission(X1, c1_1, c2_1, r1_1, r2_1, w1_1)
        u2 = _get_emission(X2, c1_2, c2_2, r1_2, r2_2, w1_2)
        _, rewards, _, _, _ = env.step(np.array([u1, u2], dtype=np.int64))
        total -= rewards
    return total


# ── Non-domination filter (minimization form) ─────────────────────────────────
def _is_nondominated(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)
    if _moocore_is_nd is not None:
        return _moocore_is_nd(arr)
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


def _eval_one_policy(
        i: int,
        row: pd.Series,
        policy_kind: str,
        eval_scenarios: np.ndarray,
        n_obj: int,
) -> Tuple[int, np.ndarray]:
    """Evaluate one policy across all eval scenarios; return (index, mean)."""
    n_scen = len(eval_scenarios)
    returns = np.zeros((n_scen, n_obj), dtype=np.float64)
    for s in range(n_scen):
        env = build_env(eval_scenarios[s], n_obj)
        if policy_kind == 'intertemporal':
            returns[s] = rollout_intertemporal(env, row)
        elif policy_kind == 'dps':
            returns[s] = rollout_dps(env, row)
        else:
            raise ValueError(f'Unknown policy_kind: {policy_kind!r}')
    return i, returns.mean(axis=0)


# ── Cell spec + per-cell evaluation ───────────────────────────────────────────
@dataclass
class CellSpec:
    folder_path: str
    policy_kind: str
    algo: str
    method: str
    n_obj: int
    archive_files: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    archive_stem: Optional[str] = None
    archive_nfe: Optional[str] = None


def evaluate_cell(
        spec: CellSpec,
        eval_scenarios: np.ndarray,
        policy_workers: int,
) -> dict:
    t0 = time.monotonic()

    # 1. Read and merge archives; detect decision columns from the first file.
    first_df = pd.read_csv(spec.archive_files[0][0])
    dec_cols = detect_dec_cols(first_df)

    parts = []
    for fpath, _ in spec.archive_files:
        df = pd.read_csv(fpath)
        parts.append(df[dec_cols])

    merged = pd.concat(parts, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=dec_cols, ignore_index=True)
    n_after = len(merged)

    # 2. Evaluate policies across a thread pool.
    n_policies = len(merged)
    means = np.zeros((n_policies, spec.n_obj), dtype=np.float64)

    futures_map: dict = {}
    with ThreadPoolExecutor(max_workers=policy_workers) as executor:
        for i, row in merged.iterrows():
            fut = executor.submit(
                _eval_one_policy,
                i, row, spec.policy_kind, eval_scenarios, spec.n_obj)
            futures_map[fut] = i
        for fut in as_completed(futures_map):
            idx, row_means = fut.result()
            means[idx] = row_means

    # 3. Pareto filter (minimization form).
    nd_mask = _is_nondominated(means)
    n_dom_dropped = int((~nd_mask).sum())

    out = merged[nd_mask].reset_index(drop=True).copy()
    means_nd = means[nd_mask]
    for j in range(spec.n_obj):
        out[f'o{j + 1}_mean'] = means_nd[:, j]
    out.insert(0, 'policy_id', np.arange(len(out)))

    # 4. Save.
    out_name = f'archives_{spec.archive_stem}_{spec.archive_nfe}_evaluated.csv'
    out.to_csv(os.path.join(spec.folder_path, out_name), index=False)

    elapsed = time.monotonic() - t0
    return {
        'folder': os.path.basename(spec.folder_path),
        'n_files': len(spec.archive_files),
        'n_before': n_before,
        'n_after_dedup': n_after,
        'n_dup': n_before - n_after,
        'n_nd': len(out),
        'n_dom': n_dom_dropped,
        'out_name': out_name,
        'elapsed_s': elapsed,
    }


def _cell_worker(args: tuple) -> dict:
    spec, eval_scenarios, policy_workers = args
    try:
        return evaluate_cell(spec, eval_scenarios, policy_workers)
    except Exception:
        return {
            'folder': os.path.basename(spec.folder_path),
            'error': traceback.format_exc(),
        }


# ── Cell discovery + walker ───────────────────────────────────────────────────
def discover_cells(base: str) -> List[CellSpec]:
    cells = []
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        policy_kind, algo, method, n_obj_str, _seed = m.groups()
        n_obj = int(n_obj_str)
        folder_path = os.path.join(base, d)
        if not os.path.isdir(folder_path):
            continue

        archive_files = []
        archive_stem = None
        archive_nfe = None
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith('_evaluated.csv'):
                continue
            if not fname.startswith('archives_'):
                continue
            am = ARCHIVE_RE.match(fname)
            if not am or '_pruned' in fname:
                continue
            stem, nfe, ref_num = am.groups()
            ref_num = int(ref_num) if ref_num is not None else None
            archive_files.append((os.path.join(folder_path, fname), ref_num))
            archive_stem = stem
            archive_nfe = nfe

        if not archive_files:
            continue

        cells.append(CellSpec(
            folder_path=folder_path,
            policy_kind=policy_kind,
            algo=algo,
            method=method,
            n_obj=n_obj,
            archive_files=archive_files,
            archive_stem=archive_stem,
            archive_nfe=archive_nfe,
        ))
    return cells


def walk_and_evaluate(
        base: str,
        eval_scenarios_path: str,
        cell_workers: int = 0,
        policy_workers: int = 4,
        dry_run: bool = False,
) -> None:
    eval_scenarios = np.load(eval_scenarios_path)
    print(f'Loaded {len(eval_scenarios)} evaluation scenarios '
          f'from {eval_scenarios_path}')

    cells = discover_cells(base)
    print(f'Discovered {len(cells)} cells to evaluate.')

    if dry_run:
        for spec in cells:
            print(f'  [{spec.policy_kind}] '
                  f'{os.path.basename(spec.folder_path)}'
                  f' - {len(spec.archive_files)} file(s)')
        return

    n_workers = cell_workers or os.cpu_count() or 4
    n_workers = min(n_workers, len(cells)) if cells else 1
    print(f'Processing with up to {n_workers} cell worker(s), '
          f'{policy_workers} policy thread(s) each.\n')

    work_items = [(spec, eval_scenarios, policy_workers) for spec in cells]

    t_start = time.monotonic()
    n_done = 0
    n_errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cell_worker, item): item[0]
                   for item in work_items}
        for fut in as_completed(futures):
            result = fut.result()
            n_done += 1
            if 'error' in result:
                n_errors += 1
                print(f'  ERROR in {result["folder"]}:\n{result["error"]}',
                      file=sys.stderr)
            else:
                r = result
                print(f'  [{n_done:2d}/{len(cells)}] {r["folder"]}: '
                      f'{r["n_files"]} file(s), '
                      f'{r["n_before"]} -> {r["n_after_dedup"]} '
                      f'(-{r["n_dup"]} dup) -> {r["n_nd"]} ND '
                      f'(-{r["n_dom"]} dom) | {r["elapsed_s"]:.1f}s '
                      f'-> {r["out_name"]}')

    elapsed = time.monotonic() - t_start
    status = 'with errors' if n_errors else 'successfully'
    print(f'\nDone {status}. {n_done - n_errors}/{len(cells)} cells written '
          f'in {elapsed:.1f}s.')


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', default='../constrained_lake_data/robust/moea',
                   help='Root folder containing the cell subdirectories')
    p.add_argument('--eval-scenarios',
                   default='../lakes/lake_scenarios_eval.npy',
                   dest='eval_scenarios',
                   help='Path to the structured numpy array of eval scenarios')
    p.add_argument('--workers', type=int, default=0,
                   help='Cell-level worker processes (0 = os.cpu_count())')
    p.add_argument('--policy-workers', type=int, default=4,
                   dest='policy_workers',
                   help='Threads per cell for policy evaluation (default: 4)')
    p.add_argument('--dry-run', action='store_true',
                   help='Discover and list cells without evaluating')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    walk_and_evaluate(
        base=args.base,
        eval_scenarios_path=args.eval_scenarios,
        cell_workers=args.workers,
        policy_workers=args.policy_workers,
        dry_run=args.dry_run,
    )
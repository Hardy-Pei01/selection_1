from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from two_lake import TwoLakeEnv
from morl.pql import PQL

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None

FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(multi|moro)_(\d+)$'
)
AGENT_RE = re.compile(r'^agent_(.+?)_(\d+)(?:_(\d+))?\.pkl$')

MAX_POLICIES_PER_AGENT = 500


def _build_q_cache(agent, decomp):
    cache = {}
    gamma = agent.gamma
    nd_dict = agent.nd_decomp if decomp else agent.non_dominated
    for state, counts in agent.counts.items():
        per_action = {}
        for a in range(agent.num_actions):
            if counts[a] == 0:
                continue
            nd_set = nd_dict[state][a]
            if not nd_set:
                continue
            Q = np.array(list(nd_set), dtype=float)
            im_rew = agent.avg_reward[state][a]
            Qsa = gamma * Q + im_rew
            per_action[a] = (Q, Qsa)
        if per_action:
            cache[state] = per_action
    return cache


def _pick_action_cached(cache, state_flat, target):
    per_action = cache.get(state_flat)
    if per_action is None:
        return 0, target, False

    best_action = None
    best_dist = np.inf
    next_target = target
    for a, (Q, Qsa) in per_action.items():
        dists = np.abs(Qsa - target).sum(axis=1)
        i = int(np.argmin(dists))
        if dists[i] < best_dist:
            best_dist = float(dists[i])
            best_action = a
            next_target = Q[i]

    if best_action is None:
        return 0, target, False
    return best_action, next_target, True


def _rollout_qtable(cache, env, target_vec, n_obj, env_shape, action_nvec):
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj, dtype=np.float64)

    for _ in range(env.n_gym_steps):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        best_action, next_target, _ = _pick_action_cached(cache, state_flat, target)
        action_nd = np.unravel_index(best_action, action_nvec)
        obs, reward, terminated, truncated, _ = env.step(np.array(action_nd, dtype=np.int64))
        total += np.asarray(reward, dtype=np.float64)
        target = next_target
        if terminated or truncated:
            break

    return total


def _is_nondominated(values):
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


def build_env(scenario, n_obj):
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


def _load_agent(agent_path, n_obj):
    with open(agent_path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios', None) if saved_robust else None

    env = TwoLakeEnv(num_obj=n_obj)
    kwargs = dict(env=env, ref_point=np.array(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        kwargs['robust'] = True
        kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**kwargs)
    agent.load_q_table(agent_path)
    return agent


@dataclass
class CellSpec:
    folder_path: str
    scoring: str
    method: str
    n_obj: int
    agent_files: List[Tuple[str, Optional[int]]] = field(default_factory=list)
    agent_stem: Optional[str] = None
    agent_nfe: Optional[str] = None


def _eval_one_policy_batch(args):
    (target_vecs, scenarios_data, n_obj, env_shape_tuple,
     action_nvec, cache_serializable) = args

    # Reconstruct cache from serializable form (numpy arrays preserved)
    cache = cache_serializable

    # Build envs from scenario data
    envs = []
    for s in scenarios_data:
        envs.append(TwoLakeEnv(
            b1=s['b1'], q1=s['q1'], b2=s['b2'], q2=s['q2'],
            inflow_seed1=s['inflow_seed1'], inflow_seed2=s['inflow_seed2'],
            Pcrit1=s['Pcrit1'], Pcrit2=s['Pcrit2'],
            num_obj=n_obj,
        ))

    env_shape = np.array(env_shape_tuple)

    results = []
    for target_vec in target_vecs:
        scenario_returns = np.zeros((len(envs), n_obj), dtype=np.float64)
        for s_idx, env in enumerate(envs):
            scenario_returns[s_idx] = _rollout_qtable(
                cache, env, target_vec, n_obj, env_shape, action_nvec,
            )
        results.append((scenario_returns.mean(axis=0), tuple(float(t) for t in target_vec)))
    return results


def evaluate_cell(spec, eval_scenarios, policy_workers=1):
    t0 = time.monotonic()

    # Pre-extract scenario fields into plain dicts so they pickle cheaply.
    scenarios_data = [
        {
            'b1': float(s['b1']), 'q1': float(s['q1']),
            'b2': float(s['b2']), 'q2': float(s['q2']),
            'inflow_seed1': int(s['inflow_seed1']),
            'inflow_seed2': int(s['inflow_seed2']),
            'Pcrit1': float(s['Pcrit1']), 'Pcrit2': float(s['Pcrit2']),
        }
        for s in eval_scenarios
    ]

    all_records = []

    for fpath, ref_num in spec.agent_files:
        agent = _load_agent(fpath, n_obj=spec.n_obj)
        decomp = (agent.action_eval == 'decomposition')
        cache = _build_q_cache(agent, decomp)
        env_shape = tuple(int(x) for x in agent.env_shape)
        action_nvec = tuple(int(x) for x in agent.env.action_space.nvec)

        archive = list(agent.archive)
        if not archive:
            continue

        if MAX_POLICIES_PER_AGENT is not None and len(archive) > MAX_POLICIES_PER_AGENT:
            n_before_sub = len(archive)
            kept = agent._subsample_nd(set(archive), target_size=MAX_POLICIES_PER_AGENT)
            archive = list(kept)
            print(
                f'    [{os.path.basename(spec.folder_path)} ref={ref_num}] '
                f'subsampled archive: {n_before_sub} → {len(archive)}',
                flush=True,
            )

        if policy_workers <= 1:
            # In-process sequential rollouts
            envs = [build_env(eval_scenarios[s], spec.n_obj) for s in range(len(eval_scenarios))]
            env_shape_arr = np.array(env_shape)
            for target_vec in archive:
                scenario_returns = np.zeros((len(envs), spec.n_obj), dtype=np.float64)
                for s_idx, env in enumerate(envs):
                    scenario_returns[s_idx] = _rollout_qtable(
                        cache, env, target_vec, spec.n_obj, env_shape_arr, action_nvec,
                    )
                mean_pos = scenario_returns.mean(axis=0)
                all_records.append({
                    'agent_ref': ref_num if ref_num is not None else -1,
                    'target_vec': tuple(float(t) for t in target_vec),
                    'mean_pos': mean_pos,
                })
        else:
            # Split archive into batches, run in process pool
            batch_size = max(1, len(archive) // (policy_workers * 4))
            batches = [archive[i:i + batch_size] for i in range(0, len(archive), batch_size)]
            args_list = [
                (batch, scenarios_data, spec.n_obj, env_shape, action_nvec, cache)
                for batch in batches
            ]
            with ProcessPoolExecutor(max_workers=policy_workers) as pool:
                for results in pool.map(_eval_one_policy_batch, args_list):
                    for mean_pos, tgt_tuple in results:
                        all_records.append({
                            'agent_ref': ref_num if ref_num is not None else -1,
                            'target_vec': tgt_tuple,
                            'mean_pos': mean_pos,
                        })

    if not all_records:
        return {
            'folder': os.path.basename(spec.folder_path),
            'n_files': len(spec.agent_files),
            'n_total': 0, 'n_nd': 0, 'n_dom': 0,
            'out_name': None, 'elapsed_s': time.monotonic() - t0,
        }

    n_total = len(all_records)

    means_min = np.array([-r['mean_pos'] for r in all_records])
    nd_mask = _is_nondominated(means_min)
    n_dom = int((~nd_mask).sum())

    rows = []
    for keep, rec in zip(nd_mask, all_records):
        if not keep:
            continue
        row = {}
        if spec.method == 'multi':
            row['agent_ref'] = rec['agent_ref']
        for j in range(spec.n_obj):
            row[f'target_o{j + 1}'] = rec['target_vec'][j]
        for j in range(spec.n_obj):
            row[f'o{j + 1}_mean'] = -rec['mean_pos'][j]
        rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(0, 'policy_id', np.arange(len(out)))

    out_name = f'archives_{spec.agent_stem}_{spec.agent_nfe}_evaluated.csv'
    out_path = os.path.join(spec.folder_path, out_name)
    out.to_csv(out_path, index=False)

    return {
        'folder': os.path.basename(spec.folder_path),
        'n_files': len(spec.agent_files),
        'n_total': n_total, 'n_nd': len(out), 'n_dom': n_dom,
        'out_name': out_name, 'elapsed_s': time.monotonic() - t0,
    }


def _cell_worker(args):
    spec, eval_scenarios, policy_workers = args
    try:
        return evaluate_cell(spec, eval_scenarios, policy_workers=policy_workers)
    except Exception:
        return {
            'folder': os.path.basename(spec.folder_path),
            'error': traceback.format_exc(),
        }


def discover_cells(base):
    cells = []
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        scoring, method, n_obj = m.groups()
        n_obj = int(n_obj)
        folder_path = os.path.join(base, d)

        agent_files = []
        agent_stem = None
        agent_nfe = None
        for fname in sorted(os.listdir(folder_path)):
            am = AGENT_RE.match(fname)
            if not am:
                continue
            stem, nfe, ref_num = am.groups()
            ref_num = int(ref_num) if ref_num is not None else None
            agent_files.append((os.path.join(folder_path, fname), ref_num))
            agent_stem = stem
            agent_nfe = nfe

        if not agent_files:
            continue

        cells.append(CellSpec(
            folder_path=folder_path,
            scoring=scoring,
            method=method,
            n_obj=n_obj,
            agent_files=agent_files,
            agent_stem=agent_stem,
            agent_nfe=agent_nfe,
        ))
    return cells


def walk_and_evaluate(base, eval_scenarios_path, cell_workers=0,
                      policy_workers=1, dry_run=False):
    eval_scenarios = np.load(eval_scenarios_path)
    print(f'Loaded {len(eval_scenarios)} evaluation scenarios from {eval_scenarios_path}')

    cells = discover_cells(base)
    print(f'Discovered {len(cells)} cells.')

    if dry_run:
        for spec in cells:
            n_arch = sum(1 for _ in spec.agent_files)
            print(f'  [{spec.scoring}_{spec.method}_{spec.n_obj}] '
                  f'{os.path.basename(spec.folder_path)} — {n_arch} agent(s)')
        return

    n_workers = cell_workers or os.cpu_count() or 4
    n_workers = min(n_workers, len(cells))
    print(f'Using {n_workers} cell worker(s), {policy_workers} policy worker(s) per cell.\n')

    work_items = [(spec, eval_scenarios, policy_workers) for spec in cells]

    t_start = time.monotonic()
    n_done = 0
    n_errors = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cell_worker, item): item[0] for item in work_items}
        for fut in as_completed(futures):
            result = fut.result()
            n_done += 1
            if 'error' in result:
                n_errors += 1
                print(f'  ERROR in {result["folder"]}:\n{result["error"]}', file=sys.stderr)
            else:
                r = result
                print(
                    f'  [{n_done:2d}/{len(cells)}] {r["folder"]}: '
                    f'{r["n_files"]} agent(s), {r["n_total"]} policies → '
                    f'{r["n_nd"]} ND (−{r["n_dom"]} dom) | {r["elapsed_s"]:.1f}s '
                    f'→ {r["out_name"]}'
                )

    elapsed = time.monotonic() - t_start
    print(f'\nDone. {n_done - n_errors}/{len(cells)} cells written in {elapsed:.1f}s.')


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='../lake_data/morl_robust')
    p.add_argument('--eval-scenarios', default='../lakes/lake_scenarios_eval.npy',
                   dest='eval_scenarios')
    p.add_argument('--workers', type=int, default=0,
                   help='Cell-level worker processes (0 = os.cpu_count())')
    p.add_argument('--policy-workers', type=int, default=1, dest='policy_workers',
                   help='Process pool size *within* each cell for policy '
                        'evaluation. Default 1 (sequential within cell). '
                        'Increase to spread one big cell across cores when '
                        'a few cells dominate runtime.')
    p.add_argument('--dry-run', action='store_true')
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

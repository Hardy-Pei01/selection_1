"""Re-evaluate robust MORL (PQL) tree policies (multi and moro) on a held-out
evaluation scenario set.

A PQL Q-table does not encode a single policy — it encodes a family of
policies, one per vector in `agent.archive` (the start-state Pareto coverage
set). Each archive vector is a "target"; following it through the Q-table via
target-vector tracking induces one deterministic policy.

Target-vector tracking (per evaluation scenario):
  1. target := archive_vector; state := start state.
  2. At each step, for every visited action a, the stored set
     non_dominated[s][a] holds raw future-return vectors Q. The realised
     estimate is Qsa = gamma * Q + avg_reward[s][a]. Pick the (action, row)
     whose Qsa is closest (L1) to the current target.
  3. Take that action in the env (the eval scenario's slip pattern may flip
     it), collect the real reward, advance.
  4. target := Q[row]  — the raw future-return vector of the matched row is,
     by construction, the remaining-return target for the next state. (This
     is the canonical PQL recipe; it works regardless of gamma or whether
     intermediate rewards are zero — both of which break a
     `(target - reward) / gamma` update, e.g. the fruit tree has zero
     internal-node rewards and gamma=1.)
  5. Sum the real rewards over the episode = this policy's return on this
     scenario.

Multi: each cell trained one agent per reference scenario (ref_num 0..N).
       Every (agent_ref, archive_vector) pair is a distinct policy. Evaluate
       all of them, merge, drop dominated, write one `*_evaluated.csv` per
       seed folder.
MORO:  each cell trained one agent. Evaluate every archive vector, drop
       dominated, write one `*_evaluated.csv` per seed folder.

Output objective columns o1_mean..on_mean are in MIN-form (negated rewards),
matching the sign convention of the MOEA archives so both paradigms can be
compared and plotted with the same code.

gamma is read from each agent's saved config — PQL.load_q_table restores it
(training used gamma_tree=1.0).

Usage:
    python evaluate_tree_morl_robust.py
"""
import os
import re
import gzip
import pickle
import time

import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.pql import PQL

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None


# ── Configuration ─────────────────────────────────────────────────────────────
MORL_BASE = '../tree_data/tree_robust/morl'
EVAL_PATTERNS_PATH = '../trees/slip_patterns_depth9_eval.npy'
CSV_PATH_DIM = {
    2: '../trees/depth9_dim2.csv',
    6: '../trees/depth9_dim6.csv',
}
TREE_DEPTH = 9

# Cell folder -> (scoring, method, n_obj). Only multi/moro are robust.
FOLDER_RE = re.compile(r'^(pareto|indicator|decomposition)_(multi|moro)_(\d+)$')
# Agent files: agent_{stem}_{nfe}[_{ref_num}].pkl
AGENT_RE = re.compile(r'^agent_(.+?)_(\d+)(?:_(\d+))?\.pkl$')


# ── Pareto filter (minimization form) ─────────────────────────────────────────
def _filter_non_dominated_min(values):
    """Boolean mask of non-dominated rows. Inputs are minimization objectives
    (smaller is better — matches the negative-reward convention used by the
    MOEA archives).
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


# ── Agent loading ─────────────────────────────────────────────────────────────
def load_agent(agent_path, n_obj, csv_path):
    """Construct a PQL agent and restore its Q-table from disk.

    Learning-time parameters (gamma, ref_point, ...) are restored by
    PQL.load_q_table from the saved config — no manual override needed.
    PQL.load_q_table also handles both gzip and plain-pickle payloads.

    n_scenarios is the one parameter the constructor needs upfront for a
    robust agent (it sizes the per-scenario structures), so we peek at the
    saved config first — but only to size the constructor; the actual data
    is restored by load_q_table.
    """
    with open(agent_path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(agent_path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios') if saved_robust else None

    env = FruitTreeEnv(
        depth=TREE_DEPTH, reward_dim=n_obj, csv_path=csv_path,
        observe=True, scenario_index=None, slip_patterns_path=None,
    )
    agent_kwargs = dict(env=env, ref_point=np.asarray(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        agent_kwargs['robust'] = True
        agent_kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**agent_kwargs)
    agent.load_q_table(agent_path)
    return agent


# ── Q-cache: per-(state, action) arrays for fast rollout ──────────────────────
def build_q_cache(agent, decomp):
    """For each visited (state, action), precompute:
        cache[state][action] = (Q, Qsa)
    where Q is the raw nd-set as an (k, n_obj) array and
    Qsa = gamma * Q + avg_reward[s][a] is the realised-return estimate.

    Done once per agent; reused across every (target, scenario) rollout —
    avoids re-converting python sets to arrays at every step.
    """
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


def pick_action_cached(cache, state_flat, target, num_actions):
    """Pick the (action, next_target) for the current target via L1 matching.

    For each visited action, find the stored Qsa row closest (L1) to the
    target; the action owning the global closest row is selected, and the
    raw Q row of that match becomes the next target. Unvisited state ->
    fall back to action 0, target unchanged, found=False.
    """
    per_action = cache.get(state_flat)
    if per_action is None:
        return 0, target, False

    best_action, best_dist, next_target = None, np.inf, target
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


# ── Rollout ───────────────────────────────────────────────────────────────────
def rollout_qtable_cached(cache, env, target_vec, depth, env_shape,
                          n_obj, num_actions):
    """Target-track one policy through one eval scenario (slip pattern already
    set on env). Returns the per-objective summed reward (MAX-form).
    """
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj)
    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action, next_target, _found = pick_action_cached(
            cache, state_flat, target, num_actions)
        obs, reward, terminal, _, _ = env.step(action)
        total += reward
        target = next_target
        if terminal:
            break
    return total


def action_sequence(cache, env, target_vec, depth, env_shape, num_actions):
    """Return the tuple of actions a policy takes under NO slip — used by the
    diagnostic to count how many distinct executable policies the archive
    actually yields.
    """
    obs, _ = env.reset()
    env._slip_pattern = np.zeros(2 ** depth - 1, dtype=bool)
    target = np.array(target_vec, dtype=float)
    actions = []
    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        action, next_target, _ = pick_action_cached(
            cache, state_flat, target, num_actions)
        actions.append(action)
        obs, _, terminal, _, _ = env.step(action)
        target = next_target
        if terminal:
            break
    return tuple(actions)


# ── Per-agent evaluation ──────────────────────────────────────────────────────
def evaluate_agent(agent, eval_patterns, n_obj, depth):
    """Evaluate every archive vector of one agent across all eval scenarios.

    Returns (targets, means_pos, diag):
      targets   : (n_pol, n_obj) the archive target vectors
      means_pos : (n_pol, n_obj) mean realised reward, MAX-form
      diag      : dict with 'n_unique_seq', 'n_archive', 'mean_corr'
    """
    decomp = (agent.action_eval == 'decomposition')
    cache = build_q_cache(agent, decomp)
    env = agent.env
    env_shape = agent.env_shape
    n_actions = agent.num_actions

    archive = [np.asarray(v, dtype=float) for v in agent.archive]
    if not archive:
        return (np.empty((0, n_obj)), np.empty((0, n_obj)),
                {'n_unique_seq': 0, 'n_archive': 0, 'mean_corr': float('nan')})

    targets = np.zeros((len(archive), n_obj))
    means_pos = np.zeros((len(archive), n_obj))
    seqs = set()
    for p, target_vec in enumerate(archive):
        # diagnostic: action sequence under no slip
        seqs.add(action_sequence(cache, env, target_vec, depth,
                                 env_shape, n_actions))
        # evaluation: mean realised return across held-out scenarios
        scenario_returns = np.zeros((len(eval_patterns), n_obj))
        for s, pat in enumerate(eval_patterns):
            env._slip_pattern = pat
            scenario_returns[s] = rollout_qtable_cached(
                cache, env, target_vec, depth, env_shape, n_obj, n_actions)
        targets[p] = target_vec
        means_pos[p] = scenario_returns.mean(axis=0)

    # diagnostic: target-vs-realised correlation, averaged over objectives
    corrs = []
    for j in range(n_obj):
        if np.std(targets[:, j]) > 1e-9 and np.std(means_pos[:, j]) > 1e-9:
            corrs.append(np.corrcoef(targets[:, j], means_pos[:, j])[0, 1])
    mean_corr = float(np.nanmean(corrs)) if corrs else float('nan')

    diag = {'n_unique_seq': len(seqs), 'n_archive': len(archive),
            'mean_corr': mean_corr}
    return targets, means_pos, diag


# ── Per-seed processing ───────────────────────────────────────────────────────
def process_seed_folder(seed_dir, scoring, method, n_obj, eval_patterns):
    """Load agent(s) in one seed folder, evaluate, merge, filter, write."""
    agent_files = []
    stem, nfe = None, None
    for fname in sorted(os.listdir(seed_dir)):
        am = AGENT_RE.match(fname)
        if not am:
            continue
        stem, nfe, ref_num = am.groups()
        ref_num = int(ref_num) if ref_num is not None else None
        agent_files.append((os.path.join(seed_dir, fname), ref_num))
    if not agent_files:
        return None

    csv_path = CSV_PATH_DIM[n_obj]

    all_targets, all_means, all_ref = [], [], []
    for fpath, ref_num in agent_files:
        agent = load_agent(fpath, n_obj=n_obj, csv_path=csv_path)
        if int(agent.num_objectives) != n_obj:
            raise ValueError(
                f'{os.path.basename(fpath)}: num_objectives '
                f'{agent.num_objectives} != folder n_obj {n_obj}')
        targets, means_pos, diag = evaluate_agent(
            agent, eval_patterns, n_obj, TREE_DEPTH)

        # DIAGNOSTIC line — visible per agent in the run log.
        ref_tag = f' ref{ref_num}' if ref_num is not None else ''
        corr_str = ('nan' if np.isnan(diag['mean_corr'])
                    else f"{diag['mean_corr']:.2f}")
        print(f"      [diag]{ref_tag}: "
              f"{diag['n_unique_seq']}/{diag['n_archive']} unique policies, "
              f"target-realised corr={corr_str}")

        all_targets.append(targets)
        all_means.append(means_pos)
        ref_id = ref_num if ref_num is not None else 0
        all_ref.append(np.full(len(targets), ref_id, dtype=int))

    targets = np.vstack(all_targets)
    means_pos = np.vstack(all_means)
    agent_ref = np.concatenate(all_ref)

    # Convert to MIN-form to match the MOEA sign convention.
    means_min = -means_pos

    # Drop dominated policies (across the merged set for multi).
    keep = _filter_non_dominated_min(means_min)
    n_total = len(means_min)
    n_dom = int((~keep).sum())

    targets_k = targets[keep]
    means_k = means_min[keep]
    ref_k = agent_ref[keep]

    # Assemble output.
    out = pd.DataFrame()
    out['policy_id'] = np.arange(int(keep.sum()))
    if method == 'multi':
        out['agent_ref'] = ref_k
    for j in range(n_obj):
        out[f'target_o{j + 1}'] = targets_k[:, j]
    for j in range(n_obj):
        out[f'o{j + 1}_mean'] = means_k[:, j]

    out_name = f'archives_{stem}_{nfe}_evaluated.csv'
    out.to_csv(os.path.join(seed_dir, out_name), index=False)
    return out_name, len(agent_files), n_total, n_dom, len(out)


# ── Walker ────────────────────────────────────────────────────────────────────
def walk_and_evaluate(base, csv_path_dim, eval_patterns_path):
    if not os.path.isdir(base):
        raise SystemExit(f'MORL base not found: {base}')
    if not os.path.exists(eval_patterns_path):
        raise SystemExit(f'eval patterns not found: {eval_patterns_path}')

    eval_patterns = np.load(eval_patterns_path)
    print(f'Loaded {len(eval_patterns)} evaluation scenarios '
          f'({eval_patterns.shape[1]} nodes each)\n')

    cells = []
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        scoring, method, n_obj = m.groups()
        cells.append((d, scoring, method, int(n_obj)))
    if not cells:
        raise SystemExit(f'No multi/moro cells found under {base}')
    print(f'Found {len(cells)} robust MORL cell(s)\n')

    t0 = time.time()
    n_seeds_done = 0
    for cell_dir, scoring, method, n_obj in cells:
        print(f'{cell_dir}  [{scoring}, {method}, {n_obj}-obj]')
        cell_path = os.path.join(base, cell_dir)
        seed_folders = sorted(
            sd for sd in os.listdir(cell_path)
            if os.path.isdir(os.path.join(cell_path, sd))
        )
        for sd in seed_folders:
            seed_dir = os.path.join(cell_path, sd)
            result = process_seed_folder(
                seed_dir, scoring, method, n_obj, eval_patterns)
            if result is None:
                print(f'    {sd}: no agent files — skipped')
                continue
            out_name, n_agents, n_total, n_dom, n_kept = result
            print(f'    {sd}: {n_agents} agent(s), {n_total} policies '
                  f'→ {n_kept} non-dominated ({n_dom} dom) → {out_name}')
            n_seeds_done += 1
        print()

    print(f'Done. {n_seeds_done} seed folder(s) written in '
          f'{time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    walk_and_evaluate(
        base=MORL_BASE,
        csv_path_dim=CSV_PATH_DIM,
        eval_patterns_path=EVAL_PATTERNS_PATH,
    )
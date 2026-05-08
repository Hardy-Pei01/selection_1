import os
import re
import pickle
import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.pql import PQL

try:
    from moocore import is_nondominated as _moocore_is_nd
except ImportError:
    _moocore_is_nd = None


# ── Folder/file pattern parsing ───────────────────────────────────────────────
FOLDER_RE = re.compile(
    r'^(pareto|indicator|decomposition)_(multi|moro)_(\d+)$'
)

# Agent files: agent_{stem}[_ref].pkl
AGENT_RE = re.compile(
    r'^agent_(.+?)_(\d+)(?:_(\d+))?\.pkl$'
)


def _build_q_cache(agent, decomp):
    """For each (state, action) with counts > 0, build:
        Q_cells[state][action] = (Q_array, Qsa_array)
    where Q_array is the stored nd_set as an (k, n_obj) numpy array, and
    Qsa_array = gamma * Q + im_rew is the reconstructed Q(s,a). Caching
    these per-agent avoids per-step set->array conversion during rollouts.
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
    """Cached version of pick_action_for_target."""
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


# ── Q-table-driven action selection (legacy path, unused after caching) ───────
def pick_action_for_target(agent, state_flat, target, decomp):
    """Non-cached fallback — see pick_action_cached for the fast path."""
    best_action = None
    best_dist = np.inf
    next_target = target
    gamma = agent.gamma

    for a in range(agent.num_actions):
        if agent.counts[state_flat][a] == 0:
            continue
        im_rew = agent.avg_reward[state_flat][a]
        nd_set = (agent.nd_decomp[state_flat][a] if decomp
                  else agent.non_dominated[state_flat][a])
        if not nd_set:
            continue
        Q = np.array(list(nd_set), dtype=float)
        Qsa = gamma * Q + im_rew
        dists = np.abs(Qsa - target).sum(axis=1)
        i = int(np.argmin(dists))
        if dists[i] < best_dist:
            best_dist = float(dists[i])
            best_action = a
            next_target = Q[i]

    found = (best_action is not None)
    if not found:
        best_action = 0
    return best_action, next_target, found


def rollout_qtable_cached(cache, env, target_vec, depth, env_shape,
                          n_obj, num_actions):
    """Run Q-table-driven rollout using a precomputed Q cache."""
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(n_obj)

    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, env_shape))
        best_action, next_target, _ = pick_action_cached(
            cache, state_flat, target, num_actions
        )
        obs, reward, terminal, _, _ = env.step(best_action)
        total += reward
        target = next_target
        if terminal:
            break

    return total


def rollout_qtable(agent, env, target_vec, depth, decomp):
    """Non-cached rollout — kept for reference. evaluate_cell uses
    rollout_qtable_cached for speed.
    """
    obs, _ = env.reset()
    target = np.array(target_vec, dtype=float)
    total = np.zeros(agent.num_objectives)

    for _ in range(depth):
        state_flat = int(np.ravel_multi_index(obs, agent.env_shape))
        best_action, next_target, _ = pick_action_for_target(
            agent, state_flat, target, decomp
        )
        obs, reward, terminal, _, _ = env.step(best_action)
        total += reward
        target = next_target
        if terminal:
            break

    return total


# ── Pareto filter (minimization form) ─────────────────────────────────────────
def _filter_non_dominated_min(values):
    """Boolean mask of non-dominated rows. Inputs are minimization
    objectives (smaller is better — matches the negative-reward convention
    used by MOEA archives).
    """
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


# ── Agent loading ─────────────────────────────────────────────────────────────
def load_agent(agent_path, n_obj, csv_path):
    """Construct a PQL agent and load the saved Q-table.
    Learning-time parameters (gamma, ref_point, etc.) are restored from
    the saved config by load_q_table itself — no manual override needed.
    """
    # Inspect the saved config to know n_scenarios (for robust agents),
    # which is the one parameter the constructor needs upfront because it
    # sizes the per-scenario data structures.
    with open(agent_path, 'rb') as f:
        payload = pickle.load(f)
    cfg = payload['config']
    saved_robust = bool(cfg.get('robust', False))
    n_scenarios = payload.get('n_scenarios', None) if saved_robust else None

    env = FruitTreeEnv(
        depth=9, reward_dim=n_obj, csv_path=csv_path,
        observe=True, scenario_index=None, slip_patterns_path=None,
    )
    agent_kwargs = dict(env=env, ref_point=np.array(cfg['ref_point']))
    if saved_robust and n_scenarios is not None:
        agent_kwargs['robust'] = True
        agent_kwargs['n_scenarios'] = n_scenarios
    agent = PQL(**agent_kwargs)
    agent.load_q_table(agent_path)
    return agent


# ── Per-cell evaluation ───────────────────────────────────────────────────────
def evaluate_cell(folder_path, scoring, method, n_obj, csv_path,
                  eval_patterns, agent_files):
    """Evaluate every policy in every agent file under all eval scenarios.
    Returns (out_df, n_target_vecs_total, n_dominated_dropped, n_unreachable_dropped).

    agent_files: list of (file_path, ref_num) tuples. moro: 1; multi: 5.
    """
    all_records = []  # one per (agent_ref, target_vec)
    n_unreachable = 0  # policies whose target_vec couldn't be tracked at any step

    for fpath, ref_num in agent_files:
        agent = load_agent(fpath, n_obj=n_obj, csv_path=csv_path)
        decomp = (agent.action_eval == 'decomposition')

        # Pre-build the Q cache for this agent — converts every (s, a)
        # nd_set from a python set to a (k, n_obj) numpy array, plus the
        # reconstructed Q(s, a) = gamma * q + im_rew. Done once per agent;
        # reused for every (target_vec, scenario) rollout.
        cache = _build_q_cache(agent, decomp)

        env = agent.env
        env_shape = agent.env_shape
        n_actions = agent.num_actions

        archive = list(agent.archive)
        if not archive:
            continue

        for target_vec in archive:
            scenario_returns = np.zeros((len(eval_patterns), n_obj))
            for s, pat in enumerate(eval_patterns):
                env._slip_pattern = pat
                scenario_returns[s] = rollout_qtable_cached(
                    cache, env, target_vec, depth=9,
                    env_shape=env_shape, n_obj=n_obj, num_actions=n_actions,
                )
            mean_pos = scenario_returns.mean(axis=0)  # positive (maximization)

            all_records.append({
                'agent_ref': ref_num if ref_num is not None else -1,
                'target_vec': tuple(float(t) for t in target_vec),
                'mean_pos': mean_pos,
            })

    if not all_records:
        return pd.DataFrame(), 0, 0, 0

    n_total = len(all_records)

    # Convert means to minimization form (negate) for Pareto filter
    means_min = np.array([-r['mean_pos'] for r in all_records])
    nd_mask = _filter_non_dominated_min(means_min)
    n_dom = int((~nd_mask).sum())

    # Build output DataFrame
    rows = []
    for keep, rec in zip(nd_mask, all_records):
        if not keep:
            continue
        row = {}
        if method == 'multi':
            row['agent_ref'] = rec['agent_ref']
        for j in range(n_obj):
            row[f'target_o{j + 1}'] = rec['target_vec'][j]
        for j in range(n_obj):
            row[f'o{j + 1}_mean'] = -rec['mean_pos'][j]  # negated
        rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(0, 'policy_id', np.arange(len(out)))
    return out, n_total, n_dom, n_unreachable


# ── Walker ────────────────────────────────────────────────────────────────────
def walk_and_evaluate(base, csv_path_dim, eval_patterns_path):
    eval_patterns = np.load(eval_patterns_path)
    print(f'Loaded {len(eval_patterns)} evaluation scenarios from {eval_patterns_path}')

    n_cells = 0
    for d in sorted(os.listdir(base)):
        m = FOLDER_RE.match(d)
        if not m:
            continue
        scoring, method, n_obj = m.groups()
        n_obj = int(n_obj)
        folder_path = os.path.join(base, d)
        csv_path = csv_path_dim[n_obj]

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

        evaluated, n_total, n_dom, _ = evaluate_cell(
            folder_path=folder_path,
            scoring=scoring, method=method,
            n_obj=n_obj, csv_path=csv_path,
            eval_patterns=eval_patterns,
            agent_files=agent_files,
        )

        out_name = f'archives_{agent_stem}_{agent_nfe}_evaluated.csv'
        out_path = os.path.join(folder_path, out_name)
        evaluated.to_csv(out_path, index=False)

        print(f'  {d}: {len(agent_files)} agent(s), '
              f'{n_total} policies, '
              f'{n_total} → {len(evaluated)} after ND filter '
              f'({n_dom} dom) → {out_name}')
        n_cells += 1

    print(f'\nDone. {n_cells} cells written.')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    BASE = '../tree_data/morl_robust'
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
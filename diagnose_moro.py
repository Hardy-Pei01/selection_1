"""
Diagnostic for MORO PCS decline. Loads a tree MORO agent.pkl and prints
plain-text statistics for four hypotheses about the decline mechanism.

Usage:
    python3 diagnose_moro.py /path/to/agent_pareto_moro_6_200000.pkl

No project imports required — works on the raw pickle payload.

Assumes:
  - Tree env (depth-9), state index = level*512 + position.
  - Action 0 -> position+0,  action 1 -> position+1   (level always +1).
  - Reward at (s,a) is value(next_state); slip flips action; non-zero
    rewards only when next_state is a leaf (level == depth).
  - Robust agent has scenario_reward_means and scenario_visit_counts
    stored at every visited "from" state.
"""

import pickle
import gzip
import sys
import numpy as np

DEPTH = 9
LEVEL_STRIDE = 512  # matches PQL ravel of obs_space high=511 in both axes


# --------------------------------------------------------------------
# Loader (handles both gzipped and plain pickles)
# --------------------------------------------------------------------

def load_payload(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------
# Tree-state helpers
# --------------------------------------------------------------------

def state_to_lp(s):
    return s // LEVEL_STRIDE, s % LEVEL_STRIDE


def lp_to_state(level, position):
    return level * LEVEL_STRIDE + position


def transition(state, action):
    """Tree transition under the agent's intended action (no slip)."""
    level, position = state_to_lp(state)
    return lp_to_state(level + 1, position + action)


# --------------------------------------------------------------------
# Q-set reconstruction without instantiating PQL
# --------------------------------------------------------------------

def get_q_array(payload, state, action, gamma):
    """avg_reward[s][a] + gamma * non_dominated[s][a]   (n,d)"""
    nd = payload["non_dominated"].get(state)
    avg = payload["avg_reward"].get(state)
    if nd is None or avg is None:
        return np.empty((0, payload["config"]["num_objectives"]))
    nd_set = nd[action]
    if not nd_set:
        return np.empty((0, payload["config"]["num_objectives"]))
    nd_arr = np.asarray(list(nd_set), dtype=float)
    return avg[action] + gamma * nd_arr


# --------------------------------------------------------------------
# Greedy Q-walk for one archive target
# --------------------------------------------------------------------

def greedy_walk(payload, target_vec, gamma):
    """
    Walk from start state choosing action whose stored Q-vector is
    closest to the running target. Returns the list of (state, action,
    chosen_q_vec) cells, length = depth.
    """
    state = 0
    target = np.asarray(target_vec, dtype=float).copy()
    path = []
    for _ in range(DEPTH):
        # candidate (action, q) pairs
        candidates = []
        for a in (0, 1):
            qa = get_q_array(payload, state, a, gamma)
            if qa.shape[0] == 0:
                continue
            for q in qa:
                candidates.append((a, q))
        if not candidates:
            break
        # nearest q to target
        best_idx = int(np.argmin([np.linalg.norm(target - q)
                                  for _, q in candidates]))
        action, qvec = candidates[best_idx]
        path.append((state, action, qvec))
        # propagate target
        avg = payload["avg_reward"][state][action]
        if gamma > 0:
            target = (target - avg) / gamma
        state = transition(state, action)
    return path


# --------------------------------------------------------------------
# Diagnostic 1: per-scenario visit count distribution at greedy cells
# --------------------------------------------------------------------

def diag_visit_distribution(payload, on_path_cells, n_scenarios):
    print("=" * 72)
    print("[1] Per-scenario visit count distribution at greedy-path cells")
    print("=" * 72)
    print(f"  n_scenarios = {n_scenarios}")
    print(f"  on-path cells = {len(on_path_cells)}")
    print()
    print(f"  {'level':>5}  {'state':>5}  {'a':>1}  "
          f"{'min':>6}  {'p20':>6}  {'med':>6}  {'mean':>7}  "
          f"{'p80':>6}  {'max':>6}  {'unvis':>5}  {'gini':>5}")
    print("  " + "-" * 70)
    for (state, action) in on_path_cells:
        svc = payload["scenario_visit_counts"][state][:, action]  # (n_s,)
        mn, p20, med, mean, p80, mx = (
            svc.min(), np.percentile(svc, 20),
            np.median(svc), svc.mean(),
            np.percentile(svc, 80), svc.max()
        )
        n_unvis = int((svc == 0).sum())
        # Gini coefficient (0 = uniform, 1 = single dominant scenario)
        sorted_v = np.sort(svc.astype(float))
        n = len(sorted_v); cum = sorted_v.cumsum()
        gini = (n + 1 - 2 * (cum.sum() / cum[-1])) / n if cum[-1] > 0 else 0.0
        level, pos = state_to_lp(state)
        print(f"  {level:>5}  {pos:>5}  {action:>1}  "
              f"{int(mn):>6d}  {int(p20):>6d}  {int(med):>6d}  "
              f"{mean:>7.1f}  {int(p80):>6d}  {int(mx):>6d}  "
              f"{n_unvis:>5d}  {gini:>5.2f}")
    print()


# --------------------------------------------------------------------
# Diagnostic 2: which scenario contributes the p20 at each cell
# --------------------------------------------------------------------

def diag_p20_contributors(payload, on_path_cells, n_scenarios):
    print("=" * 72)
    print("[2] Bottom-20% scenario set at greedy-path cells, per objective")
    print("=" * 72)
    n_obj = payload["config"]["num_objectives"]
    print(f"  n_obj = {n_obj}")
    print()

    # Collect bottom-20% scenario sets, per (cell, objective)
    bottom_sets = []  # list of (state, action, [set_per_obj])
    for (state, action) in on_path_cells:
        srm = payload["scenario_reward_means"][state][:, action, :]   # (n_s, d)
        svc = payload["scenario_visit_counts"][state][:, action]
        visited = svc > 0
        per_obj_sets = []
        for o in range(n_obj):
            vals = srm[visited, o]
            if vals.size == 0:
                per_obj_sets.append(set())
                continue
            # 20th percentile threshold (lower-is-worse for max problems)
            thresh = np.percentile(vals, 20)
            indices = np.where(visited)[0]
            bottom = set(indices[srm[visited, o] <= thresh].tolist())
            per_obj_sets.append(bottom)
        bottom_sets.append((state, action, per_obj_sets))

    # Per-cell summary
    print("  Per cell, size of bottom-20% scenario set per objective")
    print(f"  {'level':>5}  {'state':>5}  {'a':>1}  " +
          "  ".join([f"o{o}" for o in range(n_obj)]) +
          f"   {'union':>6}  {'inter':>6}")
    print("  " + "-" * 70)
    for (state, action, per_obj_sets) in bottom_sets:
        union = set().union(*per_obj_sets)
        inter = (set.intersection(*per_obj_sets)
                 if all(per_obj_sets) else set())
        sizes = [len(s) for s in per_obj_sets]
        level, pos = state_to_lp(state)
        print(f"  {level:>5}  {pos:>5}  {action:>1}  " +
              "  ".join([f"{n:>2d}" for n in sizes]) +
              f"   {len(union):>6d}  {len(inter):>6d}")

    # Stability across cells: how often does each scenario appear in
    # the bottom-20% set across all (cell, objective) pairs?
    print()
    print("  Stability: # of (cell, obj) pairs where each scenario is in p20")
    counter = np.zeros(n_scenarios, dtype=int)
    n_pairs = 0
    for (_, _, per_obj_sets) in bottom_sets:
        for s_set in per_obj_sets:
            for sc in s_set:
                counter[sc] += 1
            if s_set:
                n_pairs += 1
    print(f"  total (cell, obj) pairs with non-empty p20 set: {n_pairs}")
    if n_pairs > 0:
        print(f"  scenarios sorted by frequency (descending):")
        order = np.argsort(-counter)
        for sc in order:
            if counter[sc] > 0:
                pct = 100.0 * counter[sc] / n_pairs
                print(f"    sc {sc:>3d}:  {counter[sc]:>4d} ({pct:>5.1f}%)")
        print(f"  scenarios never in any p20 set: "
              f"{int((counter == 0).sum())}/{n_scenarios}")
    print()


# --------------------------------------------------------------------
# Diagnostic 3: start-state PCS sensitivity to aggregation
# --------------------------------------------------------------------

def _is_nd(arr):
    """Naive non-dominated mask under maximisation. Sufficient at small sizes."""
    if arr.shape[0] <= 1:
        return np.ones(arr.shape[0], dtype=bool)
    n = arr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            # j dominates i if j >= i everywhere and strictly > somewhere
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                keep[i] = False
                break
    return keep


def _hv_2d_neg(points, ref):
    """Hypervolume in maximisation form for d>=2 — naive for any d.
       Uses the inclusion-exclusion-free 'dominated volume' under
       monotone reference."""
    # Use moocore if available, else a robust loop on filtered nd front
    try:
        import moocore
        nd = points[_is_nd(points)]
        if nd.shape[0] == 0:
            return 0.0
        # moocore is min-form
        return float(moocore.hypervolume(-nd, ref=-np.asarray(ref)))
    except Exception:
        nd = points[_is_nd(points)]
        if nd.shape[0] == 0:
            return 0.0
        # 2-D fallback
        if nd.shape[1] == 2:
            order = np.argsort(-nd[:, 0])
            sorted_nd = nd[order]
            hv, prev_y = 0.0, ref[1]
            for x, y in sorted_nd:
                if x > ref[0] and y > prev_y:
                    hv += (x - ref[0]) * (y - prev_y)
                    prev_y = y
            return hv
        return float("nan")


def diag_aggregation_sensitivity(payload, gamma):
    print("=" * 72)
    print("[3] Start-state PCS sensitivity to avg_reward aggregation")
    print("=" * 72)
    print("  (recomputes avg_reward only at start state; "
          "non_dominated[next_state] held fixed)")
    print()
    n_obj = payload["config"]["num_objectives"]
    ref = np.asarray(payload["config"]["ref_point"], dtype=float)

    state = 0
    srm = payload["scenario_reward_means"][state]   # (n_s, n_a, n_obj)
    svc = payload["scenario_visit_counts"][state]   # (n_s, n_a)

    aggregations = {
        "mean": lambda v: v.mean(axis=0),
        "median": lambda v: np.median(v, axis=0),
        "p10": lambda v: np.percentile(v, 10, axis=0),
        "p20": lambda v: np.percentile(v, 20, axis=0),
        "p30": lambda v: np.percentile(v, 30, axis=0),
        "min": lambda v: v.min(axis=0),
        "max": lambda v: v.max(axis=0),
    }

    print(f"  {'aggreg':>8}  {'pcs':>5}  {'hv_max':>10}")
    print("  " + "-" * 30)
    for name, fn in aggregations.items():
        # rebuild q_set per action with replaced avg_reward at start state
        all_q = []
        for a in (0, 1):
            visited_mask = svc[:, a] > 0
            if visited_mask.sum() == 0:
                continue
            new_avg = fn(srm[visited_mask, a, :])
            nd_set = payload["non_dominated"][state][a]
            if not nd_set:
                continue
            nd_arr = np.asarray(list(nd_set), dtype=float)
            q = new_avg + gamma * nd_arr
            all_q.append(q)
        if not all_q:
            print(f"  {name:>8}  {'-':>5}  {'-':>10}")
            continue
        Q = np.vstack(all_q)
        nd_mask = _is_nd(Q)
        Q_nd = Q[nd_mask]
        hv = _hv_2d_neg(Q_nd, ref)
        print(f"  {name:>8}  {Q_nd.shape[0]:>5}  {hv:>10.4f}")
    print()


# --------------------------------------------------------------------
# Diagnostic 4: coverage gap on-path vs off-path
# --------------------------------------------------------------------

def diag_coverage_gap(payload, on_path_cells, n_scenarios):
    print("=" * 72)
    print("[4] Per-cell scenario coverage gap")
    print("=" * 72)
    n_actions = payload["config"]["num_actions"]
    on_set = set((s, a) for (s, a) in on_path_cells)

    on_full, on_low, on_total = 0, 0, 0
    off_full, off_low, off_total = 0, 0, 0

    for state, svc_mat in payload["scenario_visit_counts"].items():
        for a in range(n_actions):
            n_visited = int((svc_mat[:, a] > 0).sum())
            full = n_visited == n_scenarios
            low = n_visited < 10
            if (state, a) in on_set:
                on_total += 1
                on_full += int(full)
                on_low += int(low)
            else:
                off_total += 1
                off_full += int(full)
                off_low += int(low)

    def pct(num, den):
        return f"{(100.0 * num / den):>5.1f}%" if den else "  n/a "

    print(f"  on-path  cells: {on_total:>5d}  "
          f"all-{n_scenarios} visited: {on_full:>5d} ({pct(on_full, on_total)})  "
          f"<10 visited: {on_low:>5d} ({pct(on_low, on_total)})")
    print(f"  off-path cells: {off_total:>5d}  "
          f"all-{n_scenarios} visited: {off_full:>5d} ({pct(off_full, off_total)})  "
          f"<10 visited: {off_low:>5d} ({pct(off_low, off_total)})")

    # Distribution buckets across all visited cells
    print()
    print("  Distribution of (s,a) cells by # of scenarios visited:")
    buckets = [(0, 0), (1, 4), (5, 9), (10, 19), (20, 29),
               (30, 39), (40, 49), (50, 50)]
    counts_on = [0] * len(buckets)
    counts_off = [0] * len(buckets)
    for state, svc_mat in payload["scenario_visit_counts"].items():
        for a in range(n_actions):
            n_visited = int((svc_mat[:, a] > 0).sum())
            for bi, (lo, hi) in enumerate(buckets):
                if lo <= n_visited <= hi:
                    if (state, a) in on_set:
                        counts_on[bi] += 1
                    else:
                        counts_off[bi] += 1
                    break
    print(f"  {'bucket':>10}  {'on-path':>8}  {'off-path':>8}")
    for (lo, hi), co, cf in zip(buckets, counts_on, counts_off):
        label = f"[{lo}]" if lo == hi else f"[{lo}-{hi}]"
        print(f"  {label:>10}  {co:>8d}  {cf:>8d}")
    print()


# --------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------

def main(path):
    payload = load_payload(path)
    cfg = payload["config"]
    gamma = float(cfg["gamma"])
    n_scenarios = int(payload.get("n_scenarios",
                                  payload.get("config", {}).get("n_scenarios", 50)))

    # Header
    print()
    print("#" * 72)
    print(f"#  Diagnostic for {path}")
    print("#" * 72)
    print(f"  env_id:       {cfg.get('env_id')}")
    print(f"  num_obj:      {cfg['num_objectives']}")
    print(f"  num_actions:  {cfg['num_actions']}")
    print(f"  gamma:        {gamma}")
    print(f"  ref_point:    {cfg['ref_point']}")
    print(f"  action_eval:  {payload.get('action_eval')}")
    print(f"  global_step:  {payload.get('global_step')}")
    print(f"  archive size: {len(payload['archive'])}")
    print(f"  n_scenarios:  {n_scenarios}")
    print()

    # Build greedy paths for every archive target, union the (s,a) cells
    on_path_cells = set()
    paths_by_target = []
    for tgt in payload["archive"]:
        path = greedy_walk(payload, tgt, gamma)
        for (s, a, _) in path:
            on_path_cells.add((s, a))
        paths_by_target.append(path)

    print(f"  Greedy walks: {len(paths_by_target)} archive targets")
    print(f"  Union of on-path (s,a) cells: {len(on_path_cells)}")
    # How many distinct levels do these cells span?
    levels = sorted({state_to_lp(s)[0] for (s, a) in on_path_cells})
    print(f"  Levels covered: {levels}")
    print()

    on_path_cells = sorted(on_path_cells, key=lambda sa:
                           (state_to_lp(sa[0])[0], state_to_lp(sa[0])[1], sa[1]))

    diag_visit_distribution(payload, on_path_cells, n_scenarios)
    diag_p20_contributors(payload, on_path_cells, n_scenarios)
    diag_aggregation_sensitivity(payload, gamma)
    diag_coverage_gap(payload, on_path_cells, n_scenarios)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 diagnose_moro.py /path/to/agent.pkl")
        sys.exit(2)
    main(sys.argv[1])
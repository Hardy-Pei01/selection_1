"""MORL robust (MORO) runner — mirrors moea/moea_moro.py.

``run_moro`` is the direct analog of ``moea/moea_moro.py run_moea``:
it trains a single PQL agent whose environment cycles through a fixed set of
``n_scenarios`` sampled uncertainty scenarios on every reset, so the agent
learns a policy that is robust across the full scenario ensemble rather than
optimal for one fixed scenario.

Structural parallels with moea/moea_moro.py:
  - ``MoroFruitTreeEnv``   <-> sampling via ``build_optimization_scenarios``
  - Cycling reset()        <-> ``evaluator.robust_optimize`` over scenarios
  - ``n_scenarios=50``     <-> ``sample_uncertainties(model, 50)``
  - ``run_moro``           <-> ``run_moea`` (the moro variant)
"""

import os
import re
import time

import numpy as np
import pandas as pd

from fruit_tree import FruitTreeEnv
from morl.pql import PQL


# ── Scenario environment ──────────────────────────────────────────────────────

class MoroFruitTreeEnv(FruitTreeEnv):
    """FruitTreeEnv that trains across a fixed set of sampled slip_prob scenarios.

    At each ``reset()``, one scenario is selected uniformly at random from the
    pre-sampled set.  This mirrors what ``moea/moea_moro.py`` achieves by calling
    ``evaluator.robust_optimize`` over a set of sampled scenarios: the agent
    (policy) is evaluated—and therefore optimised—across the full ensemble
    simultaneously, rather than against a single deterministic environment.

    The scenario set is fixed at construction time (reproducible via ``seed``)
    so that comparisons across runs are fair — analogous to using the same
    ``sample_uncertainties`` call in moea/moea_moro.py.

    Args:
        depth: Tree depth.
        reward_dim: Number of objectives.
        csv_path: Path to the fruit-values CSV.
        observe: Whether the agent receives (level, position) observations.
        n_scenarios: Number of scenarios to pre-sample (default 50, matching
            ``sample_uncertainties(model, 50)`` in moea/moea_moro.py).
        seed: RNG seed for reproducible scenario sampling.
    """

    def __init__(self, depth, reward_dim, csv_path, observe,
                 n_scenarios=50, seed=42):
        rng = np.random.default_rng(seed)
        # Sample slip_prob values uniformly from [0.0, 0.2] — the same
        # range as RealParameter('slip_prob', 0.0, 0.2) in params_config.py.
        self._scenarios = rng.uniform(0.0, 0.2, n_scenarios)
        self._scenario_rng = np.random.default_rng(seed + 1)
        # Initialise the base class with the first sampled scenario.
        super().__init__(
            depth=depth,
            reward_dim=reward_dim,
            csv_path=csv_path,
            observe=observe,
            slip_prob=float(self._scenarios[0]),
        )

    def reset(self, *, seed=None, options=None):
        """Pick a random scenario then delegate to FruitTreeEnv.reset()."""
        idx = self._scenario_rng.integers(len(self._scenarios))
        self.slip_prob = float(self._scenarios[idx])
        # Derive a deterministic seed from slip_prob so each scenario is
        # internally reproducible while still varying across resets.
        return super().reset(seed=int(self.slip_prob * 1e6), options=options)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _get_depth(csv_path):
    """Extract tree depth from a CSV filename containing 'depth{d}'."""
    m = re.search(r'depth(\d+)', csv_path)
    if m:
        return int(m.group(1))
    raise ValueError(f'Cannot infer tree depth from csv_path: {csv_path}')


# ── Main runner ───────────────────────────────────────────────────────────────

def run_moro(
    scoring,
    timesteps,
    ref_point,
    n_obj,
    csv_path,
    num_weight_divisions,
    neighbourhood_size,
    output_folder,
    file_end,
    n_scenarios=50,
    start_time=None,
):
    """Train a single PQL agent robustly across all scenarios and save results.

    Direct analog of ``moea/moea_moro.py run_moea``:
    - ``MoroFruitTreeEnv`` <-> ``robust_optimize`` over sampled scenarios
    - ``n_scenarios``      <-> ``sample_uncertainties(model, 50)``
    - ``scoring``          <-> ``params.algorithm``
    - ``timesteps``        <-> ``params.nfe``

    Output files mirror moea naming:
      pcs_{file_end}.csv          <-> archives_{file_end}.csv
      convergence_{file_end}.csv  <-> convergences_{file_end}.csv

    Args:
        scoring: PQL action-evaluation method ('pareto', 'indicator',
            'decomposition').
        timesteps: Total training steps (analog of nfe).
        ref_point: Reference point for hypervolume calculation.
        n_obj: Number of objectives.
        csv_path: Path to the fruit-values CSV.
        num_weight_divisions: Weight-vector grid density (decomposition scorer).
        neighbourhood_size: Neighbourhood size for decomposition scorer.
        output_folder: Directory where results are written.
        file_end: Suffix shared by all output filenames for this experiment.
        n_scenarios: Number of scenarios to sample (default 50).
        start_time: time.time() snapshot from the caller; used to record
            wall-clock elapsed time in the convergence log.

    Returns:
        pcs_df (pd.DataFrame): The recovered robust Pareto Coverage Set.
    """
    os.makedirs(output_folder, exist_ok=True)

    depth = _get_depth(csv_path)
    env = MoroFruitTreeEnv(
        depth=depth,
        reward_dim=n_obj,
        csv_path=csv_path,
        observe=True,
        n_scenarios=n_scenarios,
    )

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay_steps=int(timesteps * 0.8),
        final_epsilon=0.05,
        num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 100),
    )

    # ── Build PCS dataframe ───────────────────────────────────────────────
    if pcs:
        pcs_arr = np.array([list(v) for v in pcs])
        pcs_df = pd.DataFrame(
            pcs_arr,
            columns=[f'o{i + 1}' for i in range(pcs_arr.shape[1])],
        )
    else:
        pcs_df = pd.DataFrame()

    # ── Build convergence dataframe ───────────────────────────────────────
    conv_df = pd.DataFrame(conv_log)

    # Attach elapsed wall-clock time — mirrors moea/moea_moro.py inside the
    # evaluator block where conv['time'] is set.
    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))

    # ── Persist ───────────────────────────────────────────────────────────
    pcs_df.to_csv(f'{output_folder}/pcs_{file_end}.csv', index=False)
    conv_df.to_csv(f'{output_folder}/convergence_{file_end}.csv', index=False)

    return pcs_df
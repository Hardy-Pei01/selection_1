"""MORL single-scenario runner — mirrors moea/single.py.

``run_morl_single`` is the direct analog of ``moea/moea_multi.py run_moea``:
it trains one PQL agent on **one** pre-configured environment (one scenario)
and saves the resulting Pareto Coverage Set (PCS) and convergence log.

The caller (``method_config.morl_single``) is responsible for building the
environment with the desired slip_prob, exactly as ``moea_multi`` is
responsible for constructing the ``Scenario`` object before calling
``single.run_moea``.
"""

import os
import time

import numpy as np
import pandas as pd

from morl.pql import PQL


def run_morl_single(
    env,
    scoring,
    timesteps,
    ref_point,
    num_weight_divisions,
    neighbourhood_size,
    output_folder,
    file_end,
    ref_num=None,
    start_time=None,
):
    """Train one PQL agent on a single scenario environment and save results.

    Direct analog of ``moea/single.py run_moea``:
    - ``env``       <-> ``model``          (the thing being optimised)
    - ``scoring``   <-> ``params.algorithm``
    - ``timesteps`` <-> ``params.nfe``
    - ``ref_num``   <-> ``ref_num``        (reference-scenario index for labelling)
    - ``start_time``<-> ``start_time``     (wall-clock elapsed time tracking)

    Output files mirror moea naming:
      pcs_{file_end}_{ref_num}.csv          <-> archives_{file_end}_{ref_num}.csv
      convergence_{file_end}_{ref_num}.csv  <-> convergences_{file_end}_{ref_num}.csv

    Args:
        env: A Gymnasium-compatible MO environment configured for one scenario.
        scoring: PQL action-evaluation method ('pareto', 'indicator',
            'decomposition').
        timesteps: Total training steps (analog of nfe).
        ref_point: Reference point for hypervolume calculation.
        num_weight_divisions: Weight-vector grid density (decomposition scorer).
        neighbourhood_size: Neighbourhood size for decomposition scorer.
        output_folder: Directory where results are written.
        file_end: Suffix shared by all output filenames for this experiment.
        ref_num: Reference-scenario index; appended to filenames and stored as
            a column in the PCS archive.  None for single (non-param) runs.
        start_time: time.time() snapshot from the caller; used to record
            wall-clock elapsed time in the convergence log.

    Returns:
        pcs_df (pd.DataFrame): The recovered Pareto Coverage Set.
    """
    os.makedirs(output_folder, exist_ok=True)

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
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

    # Attach elapsed time and reference label — mirrors moea/single.py
    if ref_num is not None:
        if start_time is not None:
            elapsed = int(time.time() - start_time)
            conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        if not pcs_df.empty:
            pcs_df['reference_scenario'] = ref_num

    # ── Persist ───────────────────────────────────────────────────────────
    suffix = f'_{ref_num}' if ref_num is not None else ''
    pcs_df.to_csv(f'{output_folder}/pcs_{file_end}{suffix}.csv', index=False)
    conv_df.to_csv(
        f'{output_folder}/convergence_{file_end}{suffix}.csv', index=False
    )

    return pcs_df
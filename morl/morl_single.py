import os
import time
import numpy as np
import pandas as pd
from morl.pql import PQL
from params_config import (
    nd_size_cap_lake, nd_update_freq_tree, nd_update_freq_lake,
    archive_cap_tree, archive_cap_lake, gamma_tree, gamma_lake,
)


def run_morl_single(
        env,
        scoring,
        timesteps,
        ref_point,
        num_weight_divisions,
        output_folder,
        file_end,
        ref_num=None,
        start_time=None,
        seed=None,
):
    os.makedirs(output_folder, exist_ok=True)

    is_tree = hasattr(env.unwrapped, 'tree_depth')
    max_nd_size = None if is_tree else nd_size_cap_lake
    max_archive_size = archive_cap_tree if is_tree else archive_cap_lake
    gamma = gamma_tree if is_tree else gamma_lake

    tag_parts = [file_end]
    if ref_num is not None:
        tag_parts.append(f'ref{ref_num}')
    tag = '_'.join(tag_parts)

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=gamma,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
        final_epsilon=0.05 if is_tree else 0.1,
        seed=seed,
        num_weight_divisions=num_weight_divisions,
        nd_update_freq=nd_update_freq_tree if is_tree else nd_update_freq_lake,
        max_nd_size=max_nd_size,
        max_archive_size=max_archive_size,
        verbose=True,
        tag=tag,
    )

    pcs, conv_log = agent.train(
        total_timesteps=timesteps,
        action_eval=scoring,
        log_every=max(1, timesteps // 10),
    )

    # ── Build PCS dataframe (snapshot of agent.archive at start state) ───
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

    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    if ref_num is not None and not pcs_df.empty:
        pcs_df['reference_scenario'] = ref_num

    # ── Persist ───────────────────────────────────────────────────────────
    # Decision sequences are not saved here; they can be re-extracted from
    # the saved agent on demand.
    suffix = f'_{ref_num}' if ref_num is not None else ''
    pcs_df.to_csv(f'{output_folder}/pcs_{file_end}{suffix}.csv', index=False)
    conv_df.to_csv(
        f'{output_folder}/convergence_{file_end}{suffix}.csv', index=False
    )
    agent.save_q_table(f'{output_folder}/agent_{file_end}{suffix}.pkl')

    return len(agent.archive)

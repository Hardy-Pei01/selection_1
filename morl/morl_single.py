import os
import time
import numpy as np
import pandas as pd
from morl.pql import PQL
from params_config import (
    nd_size_cap_lake, nd_update_freq_tree, nd_update_freq_lake,
    archive_cap_tree, archive_cap_lake,
)
from policy_eval import extract_policy, extract_lake_policy


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
    os.makedirs(output_folder, exist_ok=True)

    is_tree = hasattr(env.unwrapped, 'tree_depth')
    max_nd_size = None if is_tree else nd_size_cap_lake
    max_archive_size = archive_cap_tree if is_tree else archive_cap_lake

    tag_parts = [file_end]
    if ref_num is not None:
        tag_parts.append(f'ref{ref_num}')
    tag = '_'.join(tag_parts)

    agent = PQL(
        env=env,
        ref_point=ref_point,
        gamma=1.0,
        initial_epsilon=1.0,
        epsilon_decay_steps=timesteps,
        final_epsilon=0.05,
        num_weight_divisions=num_weight_divisions,
        neighbourhood_size=neighbourhood_size,
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

    if start_time is not None:
        elapsed = int(time.time() - start_time)
        conv_df['time'] = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    if ref_num is not None and not pcs_df.empty:
        pcs_df['reference_scenario'] = ref_num

    # ── Persist ───────────────────────────────────────────────────────────
    policy_rows = []
    for pol_id, target_vec in enumerate(pcs):
        if is_tree:
            decisions = extract_policy(agent, target_vec)
            row = {'policy_id': pol_id}
            row.update({f'l{i}': d for i, d in enumerate(decisions)})
        else:
            decisions = extract_lake_policy(agent, target_vec, env)
            row = {'policy_id': pol_id}
            for step, (u1, u2) in enumerate(decisions):
                row[f'u1_{step}'] = int(u1)
                row[f'u2_{step}'] = int(u2)
        policy_rows.append(row)

    policies_df = pd.DataFrame(policy_rows) if policy_rows else pd.DataFrame()

    if ref_num is not None:
        if not policies_df.empty:
            policies_df['reference_scenario'] = ref_num

    suffix = f'_{ref_num}' if ref_num is not None else ''
    policies_df.to_csv(f'{output_folder}/policies_{file_end}{suffix}.csv', index=False)
    pcs_df.to_csv(f'{output_folder}/pcs_{file_end}{suffix}.csv', index=False)
    conv_df.to_csv(
        f'{output_folder}/convergence_{file_end}{suffix}.csv', index=False
    )

    return policies_df
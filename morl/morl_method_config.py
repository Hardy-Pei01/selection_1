import pandas as pd

import morl.morl_single as single
import morl.morl_moro as moro
from fruit_tree import FruitTreeEnv
from params_config import default_tree_scenario, default_tree_scenario_robust, slip_patterns_path


class base_morl_params:

    def __init__(self, name, timesteps, scoring, root_folder,
                 robust):
        self.name = name
        self.timesteps = timesteps
        self.scoring = scoring
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust


class base_tree_morl_params(base_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5,
                 neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder, robust)
        self.many_obj = many_obj
        self.num_weight_divisions = num_weight_divisions
        self.neighbourhood_size = neighbourhood_size


class multi_tree_morl_params(base_tree_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, neighbourhood_size)
        # Reference scenarios spanning the slip_prob uncertainty range —
        # mirrors multi_tree_params.references in moea/moea_method_config.py.
        self.references = [
            {'scenario_index': 12},
            {'scenario_index': 24},
            {'scenario_index': 37},
            {'scenario_index': 49},
        ]


class moro_tree_morl_params(base_tree_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, neighbourhood_size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def output_file_end(params):

    return f'{params.name}_{params.timesteps}'


# ── Orchestrators ─────────────────────────────────────────────────────────────

def morl_multi(params, ref_point, n_obj, csv_path, start_time):
    depth = moro._get_depth(csv_path)
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_tree_scenario]              # {'scenario_index': None}
        label_refs = False
    else:
        refs = params.references + [default_tree_scenario_robust]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        scenario_index = ref['scenario_index']      # was ref['scenario_seed']
        print(f'  Reference scenario {ref_num} (scenario_index={scenario_index})')

        env = FruitTreeEnv(
            depth=depth, reward_dim=n_obj,
            csv_path=csv_path, observe=True,
            scenario_index=scenario_index,
            slip_patterns_path=slip_patterns_path,
        )

        policies_df = single.run_morl_single(
            env=env,
            scoring=params.scoring,
            timesteps=params.timesteps,
            ref_point=ref_point,
            num_weight_divisions=params.num_weight_divisions,
            neighbourhood_size=params.neighbourhood_size,
            output_folder=params.output_folder,
            file_end=file_end,
            ref_num=ref_num if label_refs else None,
            start_time=start_time,
        )
        archives.append(policies_df)

    # Combine all per-reference archives into one file — mirrors moea_multi
    # collecting archives across reference scenarios.
    if label_refs:
        non_empty = [p for p in archives if not p.empty]
        if non_empty:
            combined = pd.concat(non_empty, ignore_index=True)
        else:
            combined = pd.DataFrame()
        combined.to_csv(
            f'{params.output_folder}/policies_{file_end}_combined.csv', index=False
        )

    return archives


def morl_moro(params, ref_point, n_obj, csv_path, start_time):

    file_end = output_file_end(params)
    return moro.run_moro(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        csv_path=csv_path,
        num_weight_divisions=params.num_weight_divisions,
        neighbourhood_size=params.neighbourhood_size,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
    )
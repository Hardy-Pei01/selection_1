import morl.morl_single as single
import morl.morl_moro as moro
from fruit_tree import FruitTreeEnv
from two_lake import TwoLakeEnv
from params_config import default_tree_scenario, default_tree_scenario_robust, \
    slip_patterns_path, default_lake_scenario, tree_reference_scenarios, \
    lake_reference_scenarios, total_years, years_per_action


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

        self.references = tree_reference_scenarios


class moro_tree_morl_params(base_tree_morl_params):

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, neighbourhood_size)


# ── Parameter classes — lake ──────────────────────────────────────────────────

class base_lake_morl_params(base_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder, robust)
        self.many_obj = many_obj
        self.num_weight_divisions = num_weight_divisions
        self.neighbourhood_size = neighbourhood_size


class multi_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, neighbourhood_size)

        self.references = lake_reference_scenarios


class moro_lake_morl_params(base_lake_morl_params):
    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, num_weight_divisions, neighbourhood_size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def output_file_end(params):
    return f'{params.name}_{params.timesteps}'


def _build_lake_env(ref, n_obj):
    """Construct a TwoLakeEnv from a reference scenario dict."""
    return TwoLakeEnv(
        b1=ref.get('b1', 0.42), q1=ref.get('q1', 2.0),
        b2=ref.get('b2', 0.35), q2=ref.get('q2', 2.5),
        inflow_seed1=ref.get('inflow_seed1', 0),
        inflow_seed2=ref.get('inflow_seed2', 0),
        Pcrit1=ref.get('Pcrit1', None),
        Pcrit2=ref.get('Pcrit2', None),
        num_obj=n_obj,
        total_years=total_years,
        years_per_action=years_per_action,
    )


# ── Orchestrators ─────────────────────────────────────────────────────────────

def morl_multi(params, ref_point, n_obj, csv_path, start_time):
    depth = moro._get_depth(csv_path)
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_tree_scenario]
        label_refs = False
    else:
        refs = params.references + [default_tree_scenario_robust]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        scenario_index = ref['scenario_index']
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


# ── Orchestrators — lake ──────────────────────────────────────────────────────

def morl_multi_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)

    if not params.robust:
        refs = [default_lake_scenario]
        label_refs = False
    else:
        refs = params.references + [default_lake_scenario]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        print(f'  Reference scenario {ref_num} '
              f"(b1={ref.get('b1', 'default')}, q1={ref.get('q1', 'default')})")

        env = _build_lake_env(ref, n_obj)

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

    return archives


def morl_moro_lake(params, ref_point, n_obj, start_time):
    file_end = output_file_end(params)
    return moro.run_moro_lake(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        num_weight_divisions=params.num_weight_divisions,
        neighbourhood_size=params.neighbourhood_size,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
    )

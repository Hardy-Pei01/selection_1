import os
import pandas as pd
import numpy as np
from params_config import (default_tree_scenario, default_tree_scenario_robust,
                           default_lake_scenario, tree_multi_obj, tree_many_obj,
                           tree_n_scenarios)
import moea.moea_single as multi
import moea.moea_moro as moro
from ema_workbench import (Scenario)
from moea.algos import NSGAII, IBEA, MOEAD
from policy_eval import evaluate_table_archive_robust
from fruit_tree import FruitTreeEnv
from count_non_dominated import is_nondominated


class base_params(object):
    def __init__(self, name, nfe, algo, root_folder, robust, scenarios):
        self.name = name
        self.nfe = nfe
        self.algo_name = algo
        if self.algo_name == "NSGAII":
            self.algorithm = NSGAII
        elif self.algo_name == "IBEA":
            self.algorithm = IBEA
        else:
            self.algorithm = MOEAD
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.scenarios = scenarios


class base_tree_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, robust, scenarios)
        if many_obj:
            self.epsilons = [0.001] * tree_many_obj
        else:
            self.epsilons = [0.001] * tree_multi_obj


class multi_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)

        self.references = [
            {'scenario_index': 12},
            {'scenario_index': 24},
            {'scenario_index': 37},
            {'scenario_index': 49},
        ]


class moro_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)


class base_lake_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, robust, scenarios)
        if many_obj:
            self.epsilons = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]  # 6 objectives
        else:
            self.epsilons = [0.01, 0.01]  # 2 objectives


class multi_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)
        # Reference scenarios spanning the (b, q) uncertainty space.
        # Each combines a (b1,q1,b2,q2) configuration with fixed inflow seeds
        # so evaluations are fully deterministic within each reference.
        self.references = [
            {'b1': 0.10, 'q1': 2.0, 'b2': 0.10, 'q2': 2.0,
             'inflow_seed1': 1, 'inflow_seed2': 2},  # low b, low q — forgiving
            {'b1': 0.45, 'q1': 4.5, 'b2': 0.45, 'q2': 4.5,
             'inflow_seed1': 3, 'inflow_seed2': 4},  # high b, high q — sharp tipping
            {'b1': 0.42, 'q1': 2.0, 'b2': 0.35, 'q2': 2.5,
             'inflow_seed1': 5, 'inflow_seed2': 6},  # default parameters
            {'b1': 0.10, 'q1': 4.5, 'b2': 0.45, 'q2': 2.0,
             'inflow_seed1': 7, 'inflow_seed2': 8},  # mixed extremes
        ]


class moro_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)


def output_file_end(model, params):
    return f'{model.name}_{params.algo_name}_{params.nfe}'


def moea_multi(model, params, start_time, problem):
    archives = []
    convergences = []

    base = default_tree_scenario if problem == 'tree' else default_lake_scenario
    base_robust = default_tree_scenario_robust if problem == 'tree' else default_lake_scenario

    if not params.robust:
        refs = [base]
    else:
        refs = params.references + [base_robust]

    for idx, ref in enumerate(refs):
        print('Reference scenario', idx)
        file_end = output_file_end(model, params)
        results = multi.run_moea(model, params=params,
                                 file_end=file_end,
                                 reference=Scenario('reference', **ref),
                                 ref_num=idx,
                                 start_time=start_time)
        archives.append(results[0])
        convergences.append(results[1])

    if archives:
        non_empty = [a for a in archives if not a.empty]
        if non_empty:
            combined = pd.concat(non_empty, ignore_index=True)
            obj_cols = [c for c in combined.columns if c.startswith('o')]
            combined = combined.drop_duplicates(subset=obj_cols)
            # Pareto-prune the pooled set

            rewards = combined[obj_cols].values
            nd_mask = is_nondominated(np.abs(rewards))
            combined = combined[nd_mask].reset_index(drop=True)
            file_end = output_file_end(model, params)
            combined.to_csv(
                f'{params.output_folder}/archives_{file_end}_combined.csv',
                index=False
            )

    return archives, convergences


def moea_moro(model, params, start_time, problem):
    file_end = output_file_end(model, params)
    archives, convergences = moro.run_moea(model, params=params,
                                           file_end=file_end,
                                           start_time=start_time,
                                           problem=problem)

    if problem == 'tree' and 'table' in model.name.lower():
        from params_config import slip_patterns_path, tree_depth
        csv_path = next(
            (c.value for c in model.constants if c.name == 'csv_path'),
            None
        )
        if csv_path is None:
            print("  WARNING: csv_path not found in model constants, skipping pruning.")
        else:
            archive_path = f'{params.output_folder}/archives_{file_end}.csv'
            if os.path.exists(archive_path):
                env_factory = lambda idx: FruitTreeEnv(
                    depth=tree_depth, reward_dim=len(model.outcomes),
                    csv_path=csv_path, observe=True,
                    scenario_index=idx,
                    slip_patterns_path=slip_patterns_path,
                )
                pruned = evaluate_table_archive_robust(
                    archive_path=archive_path,
                    depth=tree_depth,
                    n_obj=len(model.outcomes),
                    env_factory=env_factory,
                    n_scenarios=tree_n_scenarios,
                )
                pruned.to_csv(
                    f'{params.output_folder}/archives_{file_end}_pruned.csv',
                    index=False,
                )
                print(f"  Pruned archive: {len(archives)} -> {len(pruned)} solutions")

    return archives, convergences

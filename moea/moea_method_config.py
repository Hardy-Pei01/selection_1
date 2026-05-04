import os
from params_config import (default_tree_scenario, default_tree_scenario_robust,
                           default_lake_scenario, tree_multi_obj, tree_many_obj,
                           tree_n_scenarios, tree_reference_scenarios,
                           lake_reference_scenarios)
import moea.moea_single as multi
import moea.moea_moro as moro
from ema_workbench import (Scenario)
from moea.algos import NSGAII, IBEA, MOEAD
from policy_eval import evaluate_table_archive_robust
from fruit_tree import FruitTreeEnv


class base_params(object):
    def __init__(self, name, nfe, algo, root_folder, robust):
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


class base_tree_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, robust)
        if many_obj:
            self.epsilons = [0.001] * tree_many_obj
        else:
            self.epsilons = [0.001] * tree_multi_obj


class multi_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust)

        self.references = tree_reference_scenarios


class moro_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust)


class base_lake_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, robust)
        if many_obj:
            self.epsilons = [0.1, 0.1, 0.01, 0.01, 0.01, 0.01]
        else:
            self.epsilons = [0.1, 0.01]


class multi_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust)
        self.references = lake_reference_scenarios


class moro_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust)


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

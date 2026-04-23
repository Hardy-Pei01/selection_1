from params_config import (default_tree_scenario, default_lake_scenario,
                           tree_multi_obj, tree_many_obj)
import moea.moea_multi as multi
import moea.moea_moro as moro
from ema_workbench import (Scenario)
from moea.algos import NSGAII, IBEA, MOEAD


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
            self.epsilons = [0.01] * tree_many_obj
        else:
            self.epsilons = [0.01] * tree_multi_obj


class multi_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)

        self.references = [
            {'slip_prob': 0.05},
            {'slip_prob': 0.1},
            {'slip_prob': 0.15},
            {'slip_prob': 0.2},
        ]


class moro_tree_params(base_tree_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)


class base_lake_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, robust, scenarios)
        if many_obj:
            self.epsilons = [0.1, 0.1, 0.01, 0.01, 0.01, 0.01]   # 6 objectives
        else:
            self.epsilons = [0.1, 0.01]  # 2 objectives


class multi_lake_params(base_lake_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)
        # Reference scenarios spanning the (b, q) uncertainty space.
        # Each combines a (b1,q1,b2,q2) configuration with fixed inflow seeds
        # so evaluations are fully deterministic within each reference.
        self.references = [
            {'b1': 0.10, 'q1': 2.0, 'b2': 0.10, 'q2': 2.0,
             'inflow_seed1': 1, 'inflow_seed2': 2},   # low b, low q — forgiving
            {'b1': 0.45, 'q1': 4.5, 'b2': 0.45, 'q2': 4.5,
             'inflow_seed1': 3, 'inflow_seed2': 4},   # high b, high q — sharp tipping
            {'b1': 0.42, 'q1': 2.0, 'b2': 0.35, 'q2': 2.5,
             'inflow_seed1': 5, 'inflow_seed2': 6},   # default parameters
            {'b1': 0.10, 'q1': 4.5, 'b2': 0.45, 'q2': 2.0,
             'inflow_seed1': 7, 'inflow_seed2': 8},   # mixed extremes
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
    if not params.robust:
        refs = [base]
    else:
        refs = params.references + [base]
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
    return moro.run_moea(model, params=params,
                         file_end=file_end,
                         start_time=start_time,
                         problem=problem)

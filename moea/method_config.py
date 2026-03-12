from params_config import default_scenario
import multi
import moro
from ema_workbench import (Scenario)
from algos import NSGAII, IBEA, MOEAD


class base_params(object):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        self.name = name
        self.nfe = nfe
        self.algo_name = algo
        if self.algo_name == "NSGAII":
            self.algorithm = NSGAII
            # print("NSGAII")
        elif self.algo_name == "IBEA":
            self.algorithm = IBEA
            # print("IBEA")
        else:
            self.algorithm = MOEAD
            # print("MOEAD")
        if many_obj:
            self.epsilons = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        else:
            self.epsilons = [0.1, 0.1]
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.scenarios = scenarios


class multi_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)

        self.references = \
            [{'b': 0.268340928, 'q': 3.502868198},
             {'b': 0.100879116, 'q': 3.699779508},
             {'b': 0.218652257, 'q': 2.050630370},
             {'b': 0.161967233, 'q': 3.868530616}]


class moro_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)


def output_file_end(model, params):
    return f'{model.name}_{params.algo_name}_{params.nfe}'


def moea_multi(model, params, start_time):
    archives = []
    convergences = []

    if not params.robust:
        refs = [default_scenario]
    else:
        refs = params.references[model.name] + [default_scenario]
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


def moea_moro(model, params, start_time):
    file_end = output_file_end(model, params)
    return moro.run_moea(model, params=params,
                         file_end=file_end,
                         start_time=start_time)

from moea.params_config import default_scenario
import moea.multi as multi
import moea.moro as moro
from ema_workbench import (Scenario)
from moea.algos import NSGAII, IBEA, MOEAD


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
            self.epsilons = [0.1] * 8
        else:
            self.epsilons = [0.1] * 2
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.scenarios = scenarios


class multi_params(base_params):
    def __init__(self, name, nfe, algo, root_folder, many_obj, robust, scenarios):
        super().__init__(name, nfe, algo, root_folder, many_obj, robust, scenarios)

        self.references = [
            {'slip_prob': 0.05},
            {'slip_prob': 0.1},
            {'slip_prob': 0.15},
            {'slip_prob': 0.2},
        ]


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
        refs = params.references + [default_scenario]
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

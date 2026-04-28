import os
import time

from ema_workbench import (ScalarOutcome, MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.em_framework.optimization import EpsilonProgress

import numpy as np


def percentile_20(outcomes):
    return -np.percentile(outcomes, 20)


def build_robustness_functions(num_obj):
    return [
        ScalarOutcome(
            f'p20_o{i + 1}',
            kind=ScalarOutcome.MINIMIZE,
            variable_name=f'o{i + 1}',
            function=percentile_20
        )
        for i in range(num_obj)
    ]


def build_optimization_scenarios(model, params):
    return sample_uncertainties(model, 50)


def run_moea(model, params, file_end, start_time, problem):
    archiveName = f'archives_{file_end}'
    convergenceName = f'convergences_{file_end}'

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    scenarios = build_optimization_scenarios(model, params)
    num_obj = len(model.outcomes)
    robustness_functions = build_robustness_functions(num_obj)

    with MultiprocessingEvaluator(model, n_processes=-2) as evaluator:
        arch, conv = evaluator.robust_optimize(
            robustness_functions,
            scenarios,
            algorithm=params.algorithm,
            nfe=params.nfe,
            epsilons=params.epsilons,
            convergence=[EpsilonProgress()],
            population_size=150
        )

        elapsed = int(time.time() - start_time)
        conv['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        arch.to_csv(f'{params.output_folder}/{archiveName}.csv')
        conv.to_csv(f'{params.output_folder}/{convergenceName}.csv')

        return arch, conv

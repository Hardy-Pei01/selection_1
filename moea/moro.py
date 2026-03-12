import os
import time

from ema_workbench import (Model, RealParameter, ScalarOutcome,
                           Constant, ReplicatorModel, Constraint,
                           Scenario, MultiprocessingEvaluator,
                           SequentialEvaluator)
from ema_workbench.em_framework.samplers import (sample_uncertainties,
                                                 DefaultDesigns)
from ema_workbench.em_framework.optimization import (Convergence, HyperVolume,
                                                     EpsilonProgress,
                                                     ArchiveLogger)

import pandas as pd
import numpy as np
import functools


def countRobust(robustThreshold, outcomes):
    return np.sum(outcomes >= robustThreshold) / outcomes.shape[0]


maxp = functools.partial(countRobust, 0.8)
reliability = functools.partial(countRobust, -0.99)
utility = functools.partial(countRobust, -0.8)
inertia = functools.partial(countRobust, -0.8)

robustnessFunctions = [ScalarOutcome('fraction max_P',
                                     kind=ScalarOutcome.MINIMIZE,
                                     variable_name='max_P',
                                     function=maxp),
                       ScalarOutcome('fraction reliability',
                                     kind=ScalarOutcome.MINIMIZE,
                                     variable_name='negative_utility',
                                     function=reliability),
                       ScalarOutcome('fraction inertia',
                                     kind=ScalarOutcome.MINIMIZE,
                                     variable_name='negative_inertia',
                                     function=inertia),
                       ScalarOutcome('fraction utility',
                                     kind=ScalarOutcome.MINIMIZE,
                                     variable_name='negative_utility',
                                     function=utility)]


def buildOptimizationScenarios(model, params):
    scenarios = sample_uncertainties(model, 50)
    return scenarios


def run_moea(model, params, file_end, start_time):
    archiveName = f'archives_{file_end}'
    convergenceName = f'convergences_{file_end}'

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    scenarios = buildOptimizationScenarios(model, params)

    with MultiprocessingEvaluator(model, n_processes=-2) as evaluator:
        arch, conv = evaluator.robust_optimize(
            robustnessFunctions,
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

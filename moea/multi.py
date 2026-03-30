import os
import pandas as pd
from ema_workbench import (MultiprocessingEvaluator, SequentialEvaluator,
                           save_results, load_results)
from ema_workbench.em_framework.optimization import (Convergence, HyperVolume,
                                                     EpsilonProgress,
                                                     ArchiveLogger)
import time


def run_moea(model, params, file_end, reference, ref_num, start_time):
    archiveName = f'archives_{file_end}'
    convergenceName = f'convergences_{file_end}'

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    with SequentialEvaluator(model) as evaluator:
        arch, conv = evaluator.optimize(
            algorithm=params.algorithm,
            nfe=params.nfe,
            searchover='levers',
            reference=reference,
            epsilons=params.epsilons,
            convergence=[EpsilonProgress()],
            population_size=150
        )

    if ref_num is not None:
        elapsed = int(time.time() - start_time)
        conv['time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        arch['reference_scenario'] = ref_num

    arch.to_csv(f'{params.output_folder}/{archiveName}_{ref_num}.csv')
    conv.to_csv(f'{params.output_folder}/{convergenceName}_{ref_num}.csv')

    return arch, conv

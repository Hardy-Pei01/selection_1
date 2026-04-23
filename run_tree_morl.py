import time
import numpy as np
from collections import defaultdict

from fruit_tree import FruitTreeEnv
from morl.moro import run_morl, run_moro
from morl.multi import run_multi
from moea.params_config import tree_depth, tree_multi_obj, tree_many_obj

root_folder = f'./data_morl_{tree_depth}'

run_scoring = {
    'pareto': True,
    'indicator': True,
    'decomposition': True,
}

run_scenario_method = {
    'single': True,
    'multi': False,
    'moro': False,
}

obj_uncertain = {
    'multi_obj': True,
    'many_obj': True,
}

param_uncertain = {
    'non_param': True,
    'param': False,
}

ref_points = {
    'multi_obj': np.full(tree_multi_obj, -10.0),
    'many_obj': np.full(tree_many_obj, -10.0),
}

csv_paths = {
    'multi_obj': f'./fruits/depth{tree_depth}_dim{tree_multi_obj}.csv',
    'many_obj': f'./fruits/depth{tree_depth}_dim{tree_many_obj}.csv',
}

num_objectives = {
    'multi_obj': tree_multi_obj,
    'many_obj': tree_many_obj,
}


def nested_dict():
    return defaultdict(nested_dict)


timestep_settings = nested_dict()
timestep_settings['single']['multi_obj']['non_param'] = 100000
timestep_settings['single']['many_obj']['non_param'] = 100000
timestep_settings['multi']['multi_obj']['param'] = 50000
timestep_settings['multi']['many_obj']['param'] = 50000
timestep_settings['moro']['multi_obj']['param'] = 50000
timestep_settings['moro']['many_obj']['param'] = 50000

num_weight_divisions = {
    'multi_obj': 149,
    'many_obj': 4,
}
neighbourhood_size = 10

if __name__ == '__main__':

    for key_scoring, val_scoring in run_scoring.items():
        if not val_scoring:
            continue

        for key_3, val_3 in run_scenario_method.items():
            if not val_3:
                continue

            for key_obj, val_obj in obj_uncertain.items():
                if not val_obj:
                    continue

                for key_param, val_param in param_uncertain.items():
                    if not val_param:
                        continue

                    if key_3 == 'single' and key_param == 'param':
                        continue
                    if key_3 in ('multi', 'moro') and key_param == 'non_param':
                        continue

                    name = f'{key_scoring}_{key_3}_{key_obj}_{key_param}'
                    timesteps = timestep_settings[key_3][key_obj][key_param]
                    n_obj = num_objectives[key_obj]
                    csv = csv_paths[key_obj]
                    ref = ref_points[key_obj]
                    output = f'{root_folder}/{name}'
                    file_end = f'{name}_{timesteps}'

                    print('--------------------------------------------------------------------')
                    print(f'This experiment is {name}')
                    print('--------------------------------------------------------------------')

                    start_time = time.time()

                    if key_3 == 'moro':
                        pcs_df = run_moro(
                            scoring=key_scoring, timesteps=timesteps,
                            ref_point=ref, n_obj=n_obj, csv_path=csv,
                            num_weight_divisions=num_weight_divisions[key_obj],
                            neighbourhood_size=neighbourhood_size,
                            output_folder=output, file_end=file_end,
                        )
                    elif key_3 == 'multi':
                        pcs_df = run_multi(
                            scoring=key_scoring, timesteps=timesteps,
                            ref_point=ref, n_obj=n_obj, csv_path=csv,
                            num_weight_divisions=num_weight_divisions[key_obj],
                            neighbourhood_size=neighbourhood_size,
                            output_folder=output, file_end=file_end,
                        )
                    else:
                        env = FruitTreeEnv(
                            depth=tree_depth, reward_dim=n_obj,
                            csv_path=csv, observe=True, slip_prob=0.0,
                        )
                        pcs_df = run_morl(
                            env=env, scoring=key_scoring,
                            timesteps=timesteps, ref_point=ref,
                            num_weight_divisions=num_weight_divisions[key_obj],
                            neighbourhood_size=neighbourhood_size,
                            output_folder=output, file_end=file_end,
                        )

                    elapsed = int(time.time() - start_time)
                    print(f'  Done in {time.strftime("%H:%M:%S", time.gmtime(elapsed))}. '
                          f'PCS size: {len(pcs_df)}.')

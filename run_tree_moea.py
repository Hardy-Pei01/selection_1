from ema_workbench import ema_logging
from moea.params_config import multi_objs_tree_params, many_objs_tree_params, tree_depth, tree_many_obj
from moea.method_config import multi_tree_params, moro_tree_params, moea_moro, moea_multi
from moea.model_builder import (inter_tree_model, inter_robust_tree_model, table_tree_model,
                                table_multi_objs_partially_observable_tree_model,
                                table_many_objs_partially_observable_tree_model,
                                table_robust_tree_model)
from collections import defaultdict
import time

activate_logging = True
root_folder = f'./data_{tree_depth}_{tree_many_obj}'

run_policy = {
    'intertemporal': False,
    'table': True
}
run_evo_method = {
    'NSGAII': False,
    'IBEA': False,
    'MOEAD': True
}
run_scenario_method = {
    'single': False,
    'multi': False,
    'moro': True
}

obj_uncertain = {
    'multi_obj': False,
    'many_obj': True
}

param_uncertain = {
    'non_param': False,
    'param': True
}


def tree():
    return defaultdict(tree)


nfe_settings = tree()
nfe_settings['intertemporal']['single']['multi_obj']['non_param'] = 100000
nfe_settings['intertemporal']['single']['many_obj']['non_param'] = 100000
nfe_settings['table']['single']['multi_obj']['non_param'] = 100000
nfe_settings['table']['single']['many_obj']['non_param'] = 100000
nfe_settings['intertemporal']['multi']['multi_obj']['param'] = 50000
nfe_settings['intertemporal']['multi']['many_obj']['param'] = 50000
nfe_settings['table']['multi']['multi_obj']['param'] = 50000
nfe_settings['table']['multi']['many_obj']['param'] = 50000
nfe_settings['intertemporal']['moro']['multi_obj']['param'] = 50000
nfe_settings['intertemporal']['moro']['many_obj']['param'] = 50000
nfe_settings['table']['moro']['multi_obj']['param'] = 50000
nfe_settings['table']['moro']['many_obj']['param'] = 50000

model_settings = tree()
model_settings['intertemporal']['multi_obj']['non_param'] = (inter_tree_model, 'interMulti')
model_settings['intertemporal']['many_obj']['non_param'] = (inter_tree_model, 'interMany')
model_settings['intertemporal']['multi_obj']['param'] = (inter_robust_tree_model, 'interMultiRobust')
model_settings['intertemporal']['many_obj']['param'] = (inter_robust_tree_model, 'interManyRobust')
model_settings['table']['multi_obj']['non_param'] = (table_tree_model, 'tableMulti')
model_settings['table']['many_obj']['non_param'] = (table_tree_model, 'tableMany')
model_settings['table']['multi_obj']['param'] = (table_robust_tree_model, 'tableMultiRobust')
model_settings['table']['many_obj']['param'] = (table_robust_tree_model, 'tableManyRobust')

if __name__ == '__main__':
    if (activate_logging):
        ema_logging.log_to_stderr(ema_logging.INFO)

    for key_1, value_1 in run_policy.items():
        if not value_1:
            continue

        for key_2, value_2 in run_evo_method.items():
            if not value_2:
                continue

            for key_3, value_3 in run_scenario_method.items():
                if not value_3:
                    continue

                for key_4, value_4 in obj_uncertain.items():
                    if not value_4:
                        continue

                    for key_5, value_5 in param_uncertain.items():
                        if not value_5:
                            continue

                        name = f'{key_1}_{key_2}_{key_3}_{key_4}_{key_5}'
                        nfe = nfe_settings[key_1][key_3][key_4][key_5]
                        print('--------------------------------------------------------------------')
                        print(f"This experiment is {name}")
                        print('--------------------------------------------------------------------')

                        if key_5 == "non_param":
                            robust = False
                            scenarios = None
                        elif key_5 == "param":
                            robust = True
                            scenarios = None
                        else:
                            raise Exception
                        if key_4 == "multi_obj":
                            model_params = multi_objs_tree_params
                            many_obj = False
                        elif key_4 == "many_obj":
                            model_params = many_objs_tree_params
                            many_obj = True
                        else:
                            raise Exception
                        model_func, model_name = model_settings[key_1][key_4][key_5]
                        model = model_func(model_params, model_name)
                        start_time = time.time()
                        if key_3 == "moro":
                            method_params = moro_tree_params(name=name, nfe=nfe, algo=key_2,
                                                             root_folder=root_folder, many_obj=many_obj,
                                                             robust=robust, scenarios=scenarios)
                            moea_moro(model, method_params, start_time, problem='tree')
                        else:
                            method_params = multi_tree_params(name=name, nfe=nfe, algo=key_2,
                                                              root_folder=root_folder, many_obj=many_obj,
                                                              robust=robust, scenarios=scenarios)
                            moea_multi(model, method_params, start_time, problem='tree')

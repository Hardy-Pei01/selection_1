from ema_workbench import (RealParameter, IntegerParameter, ScalarOutcome, Constant)

tree_depth = 10
tree_multi_obj = 2
tree_many_obj = 6
lake_multi_obj = 2
lake_many_obj = 6
total_years = 100
years_per_action = 5

multi_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [RealParameter('slip_prob', 0.0, 0.2)],

    'outcomes': [ScalarOutcome(f'o{i+1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_multi_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_multi_obj),
        Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_multi_obj}.csv"),
        Constant("observe", 1),
    ]
}

many_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [RealParameter('slip_prob', 0.0, 0.2)],

    'outcomes': [ScalarOutcome(f'o{i+1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_many_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_many_obj),
        Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_many_obj}.csv"),
        Constant("observe", 1),
    ]
}

default_tree_scenario = {'slip_prob': 0.0}

non_observable_constants_multi = [
    Constant("depth", tree_depth),
    Constant("num_obj", tree_multi_obj),
    Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_multi_obj}.csv"),
    Constant("observe", 0),
]

non_observable_constants_many = [
    Constant("depth", tree_depth),
    Constant("num_obj", tree_many_obj),
    Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_many_obj}.csv"),
    Constant("observe", 0),
]

# ------------------------------------------------------------------
# Two-lake model params
# ------------------------------------------------------------------
multi_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.10, 0.45),
        RealParameter('q1', 2.0,  4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0,  4.5),
        IntegerParameter('inflow_seed1', 0, 1000000),
        IntegerParameter('inflow_seed2', 0, 1000000),
    ],

    'outcomes': [ScalarOutcome(f'o{i+1}', kind=ScalarOutcome.MINIMIZE)
                 for i in range(lake_multi_obj)],

    'constants': [
        Constant("num_obj", lake_multi_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

many_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.10, 0.45),
        RealParameter('q1', 2.0,  4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0,  4.5),
        IntegerParameter('inflow_seed1', 0, 1000000),
        IntegerParameter('inflow_seed2', 0, 1000000),
    ],

    'outcomes': [ScalarOutcome(f'o{i+1}', kind=ScalarOutcome.MINIMIZE)
                 for i in range(lake_many_obj)],

    'constants': [
        Constant("num_obj", lake_many_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

default_lake_scenario = {
    'b1': 0.42, 'q1': 2.0,
    'b2': 0.35, 'q2': 2.5,
    'inflow_seed1': 0,
    'inflow_seed2': 0,
}
from ema_workbench import (RealParameter, IntegerParameter, ScalarOutcome, Constant)
from scipy.optimize import brentq


def _pcrit(b, q):
    return brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)


tree_depth = 9
tree_multi_obj = 2
tree_many_obj = 6
slip_patterns_path = f'./fruits/slip_patterns_depth{tree_depth}.npy'
tree_n_scenarios = 50
nd_size_cap_tree = 2 ** (tree_depth + 3)
nd_update_freq_tree = 1
lake_multi_obj = 2
lake_many_obj = 6
total_years = 100
years_per_action = 5
lake_scenarios_path = './lakes/lake_scenarios.npy'
lake_n_scenarios = 50
nd_size_cap_lake = 256
nd_update_freq_lake = 5

multi_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [IntegerParameter('scenario_index', 0, tree_n_scenarios - 1)],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_multi_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_multi_obj),
        Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_multi_obj}.csv"),
        Constant("observe", 1),
        Constant("slip_patterns_path", slip_patterns_path),
    ]
}

many_objs_tree_params = {
    'depth': tree_depth,

    'uncertainties': [IntegerParameter('scenario_index', 0, tree_n_scenarios - 1)],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE) for i in range(tree_many_obj)],

    'constants': [
        Constant("depth", tree_depth),
        Constant("num_obj", tree_many_obj),
        Constant("csv_path", f"./fruits/depth{tree_depth}_dim{tree_many_obj}.csv"),
        Constant("observe", 1),
        Constant("slip_patterns_path", slip_patterns_path),
    ]
}

default_tree_scenario = {'scenario_index': None}
default_tree_scenario_robust = {'scenario_index': 0}

tree_reference_scenarios = [
    {'scenario_index': 12},
    {'scenario_index': 24},
    {'scenario_index': 37},
    {'scenario_index': 49},
]

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
        RealParameter('q1', 2.0, 4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
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
        RealParameter('q1', 2.0, 4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
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
    'Pcrit1': _pcrit(0.42, 2.0),
    'Pcrit2': _pcrit(0.35, 2.5)
}

lake_reference_scenarios = [
    {'b1': 0.40, 'q1': 2.2, 'b2': 0.13, 'q2': 3.2,
     'inflow_seed1': 8240, 'inflow_seed2': 2250,
     'Pcrit1': _pcrit(0.4, 2.2), 'Pcrit2': _pcrit(0.13, 3.2)},
    {'b1': 0.15, 'q1': 2.3, 'b2': 0.22, 'q2': 3.6,
     'inflow_seed1': 8424, 'inflow_seed2': 1726,
     'Pcrit1': _pcrit(0.15, 2.3), 'Pcrit2': _pcrit(0.22, 3.6)},
    {'b1': 0.13, 'q1': 2.3, 'b2': 0.13, 'q2': 2.0,
     'inflow_seed1': 3020, 'inflow_seed2': 4330,
     'Pcrit1': _pcrit(0.13, 2.3), 'Pcrit2': _pcrit(0.13, 2.0)},
    {'b1': 0.28, 'q1': 2.8, 'b2': 0.28, 'q2': 2.9,
     'inflow_seed1': 899, 'inflow_seed2': 865,
     'Pcrit1': _pcrit(0.28, 2.8), 'Pcrit2': _pcrit(0.28, 2.9)},
]

from ema_workbench import (RealParameter, IntegerParameter, ScalarOutcome, Constant)
from scipy.optimize import brentq


def _pcrit(b, q):
    return brentq(lambda x: x ** q / (1 + x ** q) - b * x, 0.01, 1.5)


seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tree_depth = 9
tree_multi_obj = 2
tree_many_obj = 6
slip_patterns_path = f'./fruits/slip_patterns_depth{tree_depth}.npy'
tree_n_scenarios = 50
nd_size_cap_tree = 2 ** (tree_depth + 3)
nd_update_freq_tree = 1
archive_cap_tree = None
gamma_tree = 1.0
lake_multi_obj = 2
lake_many_obj = 6
total_years = 100
years_per_action = 5
lake_scenarios_path = './lakes/lake_scenarios.npy'
lake_n_scenarios = 50
nd_size_cap_lake = 128
nd_update_freq_lake = 1
archive_cap_lake = None
gamma_lake = 0.95

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
    {'b1': 0.281, 'q1': 3.368,
     'b2': 0.109, 'q2': 2.759,
     'inflow_seed1': 4783, 'inflow_seed2': 3259,
     'Pcrit1': _pcrit(0.281, 3.368),
     'Pcrit2': _pcrit(0.109, 2.759)},
    {'b1': 0.101, 'q1': 2.057,
     'b2': 0.229, 'q2': 3.198,
     'inflow_seed1': 9593, 'inflow_seed2': 6799,
     'Pcrit1': _pcrit(0.101, 2.057),
     'Pcrit2': _pcrit(0.229, 3.198)},
    {'b1': 0.180, 'q1': 3.253,
     'b2': 0.106, 'q2': 2.027,
     'inflow_seed1': 7326, 'inflow_seed2': 7581,
     'Pcrit1': _pcrit(0.180, 3.253),
     'Pcrit2': _pcrit(0.106, 2.027)},
    {'b1': 0.221, 'q1': 4.178,
     'b2': 0.395, 'q2': 2.106,
     'inflow_seed1': 895, 'inflow_seed2': 5197,
     'Pcrit1': _pcrit(0.221, 4.178),
     'Pcrit2': _pcrit(0.395, 2.106)},
]

# ────────────────────────────────────────────────────────────────────────────
# Constrained two-lake problem — parallel infrastructure
# ────────────────────────────────────────────────────────────────────────────

constrained_multi_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.10, 0.45),
        RealParameter('q1', 2.0, 4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [
        ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
        for i in range(lake_multi_obj)
    ] + [
        ScalarOutcome('n_violations_1', kind=ScalarOutcome.INFO),
    ],

    'constants': [
        Constant("num_obj", lake_multi_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

# 6-obj MOEA model params. Adds n_violations_2 (lake 2 violation count)
# as a second constraint outcome.
constrained_many_objs_lake_params = {
    'uncertainties': [
        RealParameter('b1', 0.10, 0.45),
        RealParameter('q1', 2.0, 4.5),
        RealParameter('b2', 0.10, 0.45),
        RealParameter('q2', 2.0, 4.5),
        IntegerParameter('inflow_seed1', 0, 10000),
        IntegerParameter('inflow_seed2', 0, 10000),
    ],

    'outcomes': [
        ScalarOutcome(f'o{i + 1}', kind=ScalarOutcome.MINIMIZE)
        for i in range(lake_many_obj)
    ] + [
        ScalarOutcome('n_violations_1', kind=ScalarOutcome.INFO),
        ScalarOutcome('n_violations_2', kind=ScalarOutcome.INFO),
    ],

    'constants': [
        Constant("num_obj", lake_many_obj),
        Constant("alpha", 0.4),
        Constant("delta", 0.98),
        Constant("total_years", total_years),
        Constant("years_per_action", years_per_action),
    ]
}

constrained_lake_reference_scenarios = [
    {'b1': 0.268, 'q1': 2.175,
     'b2': 0.415, 'q2': 2.073,
     'inflow_seed1': 9622, 'inflow_seed2': 1693,
     'Pcrit1': _pcrit(0.268, 2.175),
     'Pcrit2': _pcrit(0.415, 2.073)},
    {'b1': 0.222, 'q1': 4.234,
     'b2': 0.101, 'q2': 4.285,
     'inflow_seed1': 451, 'inflow_seed2': 4569,
     'Pcrit1': _pcrit(0.222, 4.234),
     'Pcrit2': _pcrit(0.101, 4.285)},
    {'b1': 0.223, 'q1': 2.245,
     'b2': 0.132, 'q2': 3.018,
     'inflow_seed1': 674, 'inflow_seed2': 1256,
     'Pcrit1': _pcrit(0.223, 2.245),
     'Pcrit2': _pcrit(0.132, 3.018)},
    {'b1': 0.434, 'q1': 2.234,
     'b2': 0.336, 'q2': 2.461,
     'inflow_seed1': 5994, 'inflow_seed2': 2608,
     'Pcrit1': _pcrit(0.434, 2.234),
     'Pcrit2': _pcrit(0.336, 2.461)},
]

from ema_workbench import (RealParameter, ScalarOutcome, Constant)

depth = 10

multi_objs_model_params = {
    'depth': depth,

    'uncertainties': [RealParameter('w', 0, 1)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", depth),
        Constant("num_obj", 2),
        Constant("csv_path", "./fruits/depth10_dim2.csv"),
        Constant("observe", 1),
    ]
}

many_objs_model_params = {
    'depth': depth,

    'uncertainties': [RealParameter('w', 0, 1)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o3', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o4', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o5', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o6', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o7', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o8', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", depth),
        Constant("num_obj", 8),
        Constant("csv_path", "./fruits/depth10_dim8.csv"),
        Constant("observe", 1),
    ]
}

default_scenario = {
    'w': 0.5
}

non_observable_constants_multi = [
    Constant("depth", depth),
    Constant("num_obj", 2),
    Constant("csv_path", "./fruits/depth10_dim2.csv"),
    Constant("observe", 0),
]

non_observable_constants_many = [
    Constant("depth", depth),
    Constant("num_obj", 8),
    Constant("csv_path", "./fruits/depth10_dim8.csv"),
    Constant("observe", 0),
]

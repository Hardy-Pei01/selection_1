from ema_workbench import (RealParameter, ScalarOutcome, Constant)

depth = 15

two_objs_model_params = {
    'depth': depth,

    'uncertainties': [RealParameter('b', 0.1, 0.45),
                      RealParameter('q', 2.0, 4.5)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", depth),
        Constant("num_obj", 2),
        Constant("csv_path", "../fruits/depth15_dim2.csv"),
        Constant("observe", 1),
    ]
}

six_objs_model_params = {
    'depth': depth,

    'uncertainties': [RealParameter('b', 0.1, 0.45),
                      RealParameter('q', 2.0, 4.5)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o3', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o4', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o5', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o6', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", depth),
        Constant("num_obj", 6),
        Constant("csv_path", "../fruits/depth15_dim6.csv"),
        Constant("observe", 1),
    ]
}

default_scenario = {
    'b': 0.42,
    'q': 2
}

non_observable_constants_2 = [
    Constant("depth", depth),
    Constant("num_obj", 2),
    Constant("csv_path", "../fruits/depth15_dim2.csv"),
    Constant("observe", 0),
]

non_observable_constants_6 = [
    Constant("depth", depth),
    Constant("num_obj", 6),
    Constant("csv_path", "../fruits/depth15_dim6.csv"),
    Constant("observe", 0),
]

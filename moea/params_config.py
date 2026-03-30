from ema_workbench import (RealParameter, ScalarOutcome, Constant)

two_objs_model_params = {
    'depth': 7,

    'uncertainties': [RealParameter('b', 0.1, 0.45),
                      RealParameter('q', 2.0, 4.5)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", 7),
        Constant("num_obj", 2),
        Constant("observe", 1),
    ]
}

six_objs_model_params = {
    'depth': 7,

    'uncertainties': [RealParameter('b', 0.1, 0.45),
                      RealParameter('q', 2.0, 4.5)],

    'outcomes': [ScalarOutcome('o1', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o2', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o3', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o4', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o5', kind=ScalarOutcome.MINIMIZE),
                 ScalarOutcome('o6', kind=ScalarOutcome.MINIMIZE)],

    'constants': [
        Constant("depth", 7),
        Constant("num_obj", 6),
        Constant("observe", 1),
    ]
}

default_scenario = {
    'b': 0.42,
    'q': 2
}
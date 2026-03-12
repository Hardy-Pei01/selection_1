from ema_workbench import (Model, IntegerParameter, RealParameter, Constant)
from fruit_tree_moea_2 import fruit_tree_inter, fruit_tree_dps, N_CENTERS


def inter_base_model(intertemporal, params):
    depth = params['depth']
    intertemporal.levers = [
        IntegerParameter('l{}'.format(i), 0, 1) for i in range(depth)
    ]
    intertemporal.constants = params['constants']
    return intertemporal


def inter_model(params, model_name):
    intertemporal = Model(model_name,
                          function=fruit_tree_inter)
    intertemporal = inter_base_model(intertemporal, params)
    intertemporal.outcomes = params['outcomes']

    return intertemporal


def inter_robust_model(params, model_name):
    intertemporal = Model(model_name,
                          function=fruit_tree_inter)
    intertemporal = inter_base_model(intertemporal, params)
    intertemporal.uncertainties = params['uncertainties']
    intertemporal.outcomes = params['outcomes']

    return intertemporal


def dps_base_model(dps, params):
    n = N_CENTERS
    levers = []
    for i in range(n):
        levers.append(RealParameter(f"c{i}_0", 0.0, 1.0))   # normalised row
        levers.append(RealParameter(f"c{i}_1", 0.0, 1.0))   # normalised pos
    for i in range(n):
        levers.append(RealParameter(f"rad{i}", 0.01, 2.0))   # radius
    for i in range(n):
        levers.append(RealParameter(f"wL{i}", 0.0, 1.0))     # left weight
    for i in range(n):
        levers.append(RealParameter(f"wR{i}", 0.0, 1.0))     # right weight

    dps.levers = levers
    dps.constants = params['constants']
    return dps


def dps_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    return dps


def dps_two_objs_partially_observable_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    dps.constants = [
        Constant("depth", 7),
        Constant("num_obj", 2),
        Constant("observe", 0),
    ]

    return dps


def dps_six_objs_partially_observable_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    dps.constants = [
        Constant("depth", 7),
        Constant("num_obj", 6),
        Constant("observe", 0),
    ]

    return dps


def dps_robust_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.uncertainties = params['uncertainties']
    dps.outcomes = params['outcomes']

    return dps
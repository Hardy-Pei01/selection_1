from ema_workbench import (Model, IntegerParameter, RealParameter, Constant)
from fruit_tree_moea import fruit_tree_inter, fruit_tree_dps
from fruit_tree_moea_robust import fruit_tree_inter_robust, fruit_tree_dps_robust
from moea.params_config import non_observable_constants_multi, non_observable_constants_many


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
                          function=fruit_tree_inter_robust)
    intertemporal = inter_base_model(intertemporal, params)
    intertemporal.uncertainties = params['uncertainties']
    intertemporal.outcomes = params['outcomes']

    return intertemporal


def dps_base_model(dps, params):
    dps.levers = [
        RealParameter('w0',        -1.0, 1.0),
        RealParameter('w1',        -1.0, 1.0),
        RealParameter('threshold',  0.0, 1.0),
    ]
    dps.constants = params['constants']
    return dps


def dps_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    return dps


def dps_multi_objs_partially_observable_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    dps.constants = non_observable_constants_multi

    return dps


def dps_many_objs_partially_observable_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps)
    dps = dps_base_model(dps, params)
    dps.outcomes = params['outcomes']

    dps.constants = non_observable_constants_many

    return dps


def dps_robust_model(params, model_name):
    dps = Model(model_name,
                function=fruit_tree_dps_robust)
    dps = dps_base_model(dps, params)
    dps.uncertainties = params['uncertainties']
    dps.outcomes = params['outcomes']

    return dps
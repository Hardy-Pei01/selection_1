from ema_workbench import (Model, IntegerParameter, RealParameter, Constant)
from moea.fruit_tree_moea import fruit_tree_inter, fruit_tree_table, fruit_tree_inter_robust, fruit_tree_table_robust
from moea.two_lake_moea import two_lake_inter, two_lake_inter_robust, two_lake_dps, two_lake_dps_robust
from moea.params_config import non_observable_constants_multi, non_observable_constants_many


def inter_base_tree_model(intertemporal, params):
    depth = params['depth']
    intertemporal.levers = [
        IntegerParameter('l{}'.format(i), 0, 1) for i in range(depth)
    ]
    intertemporal.constants = params['constants']
    return intertemporal


def inter_tree_model(params, model_name):
    intertemporal = Model(model_name,
                          function=fruit_tree_inter)
    intertemporal = inter_base_tree_model(intertemporal, params)
    intertemporal.outcomes = params['outcomes']

    return intertemporal


def inter_robust_tree_model(params, model_name):
    intertemporal = Model(model_name,
                          function=fruit_tree_inter_robust)
    intertemporal = inter_base_tree_model(intertemporal, params)
    intertemporal.uncertainties = params['uncertainties']
    intertemporal.outcomes = params['outcomes']

    return intertemporal


def table_base_tree_model(table, params):
    depth = params['depth']
    n_internal = 2 ** depth - 1
    table.levers = [
        IntegerParameter(f'n{i}', 0, 1) for i in range(n_internal)
    ]
    table.constants = params['constants']
    return table


def table_tree_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_tree_model(table, params)
    table.outcomes = params['outcomes']

    return table


def table_multi_objs_partially_observable_tree_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_tree_model(table, params)
    table.outcomes = params['outcomes']

    table.constants = non_observable_constants_multi

    return table


def table_many_objs_partially_observable_tree_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_tree_model(table, params)
    table.outcomes = params['outcomes']

    table.constants = non_observable_constants_many

    return table


def table_robust_tree_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table_robust)
    table = table_base_tree_model(table, params)
    table.uncertainties = params['uncertainties']
    table.outcomes = params['outcomes']

    return table


# ================================================================
# Two-lake builders — NEW
# ================================================================

def inter_base_lake_model(intertemporal, params):
    total_years = next(c.value for c in params['constants'] if c.name == 'total_years')
    years_per_action = next(c.value for c in params['constants'] if c.name == 'years_per_action')
    n_steps = total_years // years_per_action

    intertemporal.levers = (
            [IntegerParameter(f'u1_{i}', 0, 10) for i in range(n_steps)] +
            [IntegerParameter(f'u2_{i}', 0, 10) for i in range(n_steps)]
    )
    intertemporal.constants = params['constants']
    return intertemporal


def inter_lake_model(params, model_name):
    intertemporal = Model(model_name, function=two_lake_inter)
    intertemporal = inter_base_lake_model(intertemporal, params)
    intertemporal.outcomes = params['outcomes']
    return intertemporal


def inter_robust_lake_model(params, model_name):
    intertemporal = Model(model_name, function=two_lake_inter_robust)
    intertemporal = inter_base_lake_model(intertemporal, params)
    intertemporal.uncertainties = params['uncertainties']
    intertemporal.outcomes = params['outcomes']
    return intertemporal


def dps_base_lake_model(dps, params):
    dps.levers = [
        RealParameter("c1_1", -2, 2),
        RealParameter("c2_1", -2, 2),
        RealParameter("r1_1", 0.01, 2),
        RealParameter("r2_1", 0.01, 2),
        RealParameter("w1_1", 0, 1),
        RealParameter("c1_2", -2, 2),
        RealParameter("c2_2", -2, 2),
        RealParameter("r1_2", 0.01, 2),
        RealParameter("r2_2", 0.01, 2),
        RealParameter("w1_2", 0, 1),
    ]
    dps.constants = params['constants']
    return dps


def dps_lake_model(params, model_name):
    dps = Model(model_name,
                function=two_lake_dps)
    dps = dps_base_lake_model(dps, params)
    dps.outcomes = params['outcomes']

    return dps


def dps_robust_lake_model(params, model_name):
    dps = Model(model_name,
                function=two_lake_dps_robust)
    dps = dps_base_lake_model(dps, params)
    dps.uncertainties = params['uncertainties']
    dps.outcomes = params['outcomes']

    return dps

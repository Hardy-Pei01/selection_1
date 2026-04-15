from ema_workbench import (Model, IntegerParameter, RealParameter, Constant)
from moea.fruit_tree_moea import fruit_tree_inter, fruit_tree_table
from moea.fruit_tree_moea_robust import fruit_tree_inter_robust, fruit_tree_table_robust
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


def table_base_model(table, params):
    depth = params['depth']
    n_internal = 2 ** depth - 1
    table.levers = [
        IntegerParameter(f'n{i}', 0, 1) for i in range(n_internal)
    ]
    table.constants = params['constants']
    return table


def table_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_model(table, params)
    table.outcomes = params['outcomes']

    return table


def table_multi_objs_partially_observable_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_model(table, params)
    table.outcomes = params['outcomes']

    table.constants = non_observable_constants_multi

    return table


def table_many_objs_partially_observable_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table)
    table = table_base_model(table, params)
    table.outcomes = params['outcomes']

    table.constants = non_observable_constants_many

    return table


def table_robust_model(params, model_name):
    table = Model(model_name,
                  function=fruit_tree_table_robust)
    table = table_base_model(table, params)
    table.uncertainties = params['uncertainties']
    table.outcomes = params['outcomes']

    return table

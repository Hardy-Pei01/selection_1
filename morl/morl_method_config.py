"""MORL method configuration — mirrors moea/moea_method_config.py.

Provides:
  - Param classes that carry all hyperparameters for one experiment run,
    mirroring ``base_params``, ``base_tree_params``, ``multi_tree_params``,
    and ``moro_tree_params`` in moea/moea_method_config.py.
  - ``morl_multi``  — analog of ``moea_multi``:  handles the 'single' and
    'multi' scenario methods by running PQL on one or several reference
    scenarios.
  - ``morl_moro``   — analog of ``moea_moro``:   handles the 'moro' scenario
    method by running PQL robustly across a full scenario ensemble.

Scenario-method semantics (mirrors moea exactly):
  single  (robust=False) — one run on the base scenario (slip_prob=0.0)
  multi   (robust=True)  — one run per reference scenario + base, archives
                           combined into a single PCS file
  moro    (n/a)          — one run via MoroFruitTreeEnv cycling through
                           n_scenarios sampled scenarios

Correspondence table:
  morl                        | moea
  ----------------------------+------------------------------------------
  base_morl_params            | base_params
  base_tree_morl_params       | base_tree_params
  multi_tree_morl_params      | multi_tree_params
  moro_tree_morl_params       | moro_tree_params
  output_file_end(params)     | output_file_end(model, params)
  morl_multi(params, ...)     | moea_multi(model, params, ...)
  morl_moro(params, ...)      | moea_moro(model, params, ...)
  scoring ('pareto', ...)     | algo_name ('NSGAII', 'IBEA', 'MOEAD')
  timesteps                   | nfe
  pcs_{file_end}.csv          | archives_{file_end}.csv
  convergence_{file_end}.csv  | convergences_{file_end}.csv
"""

import pandas as pd

import morl.morl_single as single
import morl.morl_moro as moro
from fruit_tree import FruitTreeEnv
from params_config import default_tree_scenario, tree_multi_obj, tree_many_obj


# ── Parameter classes ─────────────────────────────────────────────────────────

class base_morl_params:
    """Base params shared by all MORL experiments.

    Mirrors ``base_params`` in moea/moea_method_config.py.

    Args:
        name: Experiment identifier, used as sub-folder name and filename
            prefix (analog of model.name + algo_name in moea).
        timesteps: Total PQL training steps (analog of nfe).
        scoring: PQL action-evaluation method: 'pareto', 'indicator', or
            'decomposition' (analog of algo_name / algorithm).
        root_folder: Top-level output directory; ``output_folder`` is
            ``root_folder/name``.
        robust: True for 'multi' (several reference scenarios) and 'moro'
            (robust optimisation); False for 'single' (base scenario only).
        n_scenarios: Number of scenarios sampled for MORO training
            (analog of ``sample_uncertainties(model, 50)`` in moea/moea_moro.py).
    """

    def __init__(self, name, timesteps, scoring, root_folder,
                 robust, n_scenarios=50):
        self.name = name
        self.timesteps = timesteps
        self.scoring = scoring
        self.output_folder = f'{root_folder}/{name}'
        self.robust = robust
        self.n_scenarios = n_scenarios


class base_tree_morl_params(base_morl_params):
    """Tree-specific MORL params.

    Extends ``base_morl_params`` with tree-specific hyperparameters that have
    no equivalent in the moea param classes (PQL-specific):
      - ``num_weight_divisions``: weight-vector lattice density
      - ``neighbourhood_size``:  Chebyshev neighbourhood size

    Mirrors ``base_tree_params`` in moea/moea_method_config.py (which adds
    ``epsilons``).

    Args:
        many_obj: True for the many-objective (8-dim) variant; False for the
            multi-objective (2-dim) variant.  Used by the runner to look up
            objective-count-dependent settings.
        num_weight_divisions: Passed through to PQL.
        neighbourhood_size: Passed through to PQL.
    """

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, n_scenarios=50,
                 num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         robust, n_scenarios)
        self.many_obj = many_obj
        self.num_weight_divisions = num_weight_divisions
        self.neighbourhood_size = neighbourhood_size


class multi_tree_morl_params(base_tree_morl_params):
    """Params for the 'multi' (separate single-scenario) MORL method.

    Mirrors ``multi_tree_params`` in moea/moea_method_config.py, which carries the
    list of reference scenarios.  Here, each reference is a dict with a single
    key ``slip_prob`` — the MORL analog of the dict keys used to construct
    ``Scenario`` objects in moea.

    The base scenario (slip_prob=0.0 from ``default_tree_scenario``) is always
    appended as the final reference when ``robust=True``, exactly as
    ``moea_multi`` appends ``default_tree_scenario`` to ``params.references``.
    """

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, n_scenarios=50,
                 num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, n_scenarios,
                         num_weight_divisions, neighbourhood_size)
        # Reference scenarios spanning the slip_prob uncertainty range —
        # mirrors multi_tree_params.references in moea/moea_method_config.py.
        self.references = [
            {'slip_prob': 0.05},
            {'slip_prob': 0.10},
            {'slip_prob': 0.15},
            {'slip_prob': 0.20},
        ]


class moro_tree_morl_params(base_tree_morl_params):
    """Params for the 'moro' (robust optimisation) MORL method.

    Mirrors ``moro_tree_params`` in moea/moea_method_config.py: no additional
    fields beyond the base — all MORO-specific logic lives in
    ``morl/morl_moro.py run_moro`` (and ``MoroFruitTreeEnv``), not in the params.
    """

    def __init__(self, name, timesteps, scoring, root_folder,
                 many_obj, robust, n_scenarios=50,
                 num_weight_divisions=5, neighbourhood_size=10):
        super().__init__(name, timesteps, scoring, root_folder,
                         many_obj, robust, n_scenarios,
                         num_weight_divisions, neighbourhood_size)


# ── Helpers ───────────────────────────────────────────────────────────────────

def output_file_end(params):
    """Build the shared filename suffix for one experiment.

    Mirrors ``output_file_end(model, params)`` in moea/moea_method_config.py
    (which uses model.name + algo_name + nfe).  Here, params.name already
    encodes the full experiment identity, so we only append timesteps.
    """
    return f'{params.name}_{params.timesteps}'


# ── Orchestrators ─────────────────────────────────────────────────────────────

def morl_multi(params, ref_point, n_obj, csv_path, start_time):
    """Run PQL on one or several reference scenarios and return the archives.

    Mirrors ``moea_multi`` in moea/moea_method_config.py:

    single  (robust=False):
        Runs one PQL training on the base scenario (slip_prob=0.0 from
        ``default_tree_scenario``).  The ref_num suffix is omitted from
        filenames — mirrors moea_multi passing a single base reference
        when not robust.

    multi   (robust=True):
        Loops over ``params.references + [base]``, training one PQL agent per
        reference scenario (one fixed slip_prob each).  Each run is saved
        individually (suffix ``_{ref_num}``), then all archives are combined
        into a single ``pcs_{file_end}_combined.csv`` — mirrors how
        ``moea_multi`` collects and concatenates per-reference archives.

    In both cases the environment is built here (with the reference slip_prob)
    and passed to ``single.run_morl_single``, exactly as ``moea_multi``
    constructs a ``Scenario`` object and passes it to ``single.run_moea``.

    Args:
        params: A ``multi_tree_morl_params`` (or compatible) instance.
        ref_point: PQL reference point for hypervolume.
        n_obj: Number of objectives.
        csv_path: Path to the fruit-values CSV.
        start_time: ``time.time()`` snapshot for elapsed-time logging.

    Returns:
        archives (list[pd.DataFrame]): Per-reference PCS dataframes.
    """
    depth = moro._get_depth(csv_path)
    file_end = output_file_end(params)

    base_slip = default_tree_scenario['slip_prob']  # 0.0
    if not params.robust:
        # single: one run on the base scenario — no reference index
        refs = [{'slip_prob': base_slip}]
        label_refs = False
    else:
        # multi: reference scenarios + base, each labelled by index
        refs = params.references + [{'slip_prob': base_slip}]
        label_refs = True

    archives = []
    for ref_num, ref in enumerate(refs):
        slip_prob = ref['slip_prob']
        print(f'  Reference scenario {ref_num} (slip_prob={slip_prob:.2f})')

        env = FruitTreeEnv(
            depth=depth,
            reward_dim=n_obj,
            csv_path=csv_path,
            observe=True,
            slip_prob=slip_prob,
        )

        pcs_df = single.run_morl_single(
            env=env,
            scoring=params.scoring,
            timesteps=params.timesteps,
            ref_point=ref_point,
            num_weight_divisions=params.num_weight_divisions,
            neighbourhood_size=params.neighbourhood_size,
            output_folder=params.output_folder,
            file_end=file_end,
            ref_num=ref_num if label_refs else None,
            start_time=start_time,
        )
        archives.append(pcs_df)

    # Combine all per-reference archives into one file — mirrors moea_multi
    # collecting archives across reference scenarios.
    if label_refs:
        non_empty = [p for p in archives if not p.empty]
        if non_empty:
            combined = pd.concat(non_empty, ignore_index=True)
        else:
            combined = pd.DataFrame()
        combined.to_csv(
            f'{params.output_folder}/pcs_{file_end}_combined.csv', index=False
        )

    return archives


def morl_moro(params, ref_point, n_obj, csv_path, start_time):
    """Train a single PQL agent robustly across all scenarios.

    Mirrors ``moea_moro`` in moea/moea_method_config.py: a thin wrapper that
    builds the file-end string and delegates entirely to ``moro.run_moro``,
    just as ``moea_moro`` delegates to ``moro.run_moea``.

    Args:
        params: A ``moro_tree_morl_params`` (or compatible) instance.
        ref_point: PQL reference point for hypervolume.
        n_obj: Number of objectives.
        csv_path: Path to the fruit-values CSV.
        start_time: ``time.time()`` snapshot for elapsed-time logging.

    Returns:
        pcs_df (pd.DataFrame): The recovered robust Pareto Coverage Set.
    """
    file_end = output_file_end(params)
    return moro.run_moro(
        scoring=params.scoring,
        timesteps=params.timesteps,
        ref_point=ref_point,
        n_obj=n_obj,
        csv_path=csv_path,
        num_weight_divisions=params.num_weight_divisions,
        neighbourhood_size=params.neighbourhood_size,
        n_scenarios=params.n_scenarios,
        output_folder=params.output_folder,
        file_end=file_end,
        start_time=start_time,
    )
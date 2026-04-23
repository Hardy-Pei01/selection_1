"""Pareto Q-Learning."""

import numbers
from itertools import product
from typing import Callable
from collections import defaultdict

import gymnasium as gym
import numpy as np

from morl_baselines.common.morl_algorithm import MOAgent
from morl_baselines.common.pareto import get_non_dominated
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.utils import linearly_decaying_value


def generate_weights(num_objectives: int, num_divisions: int) -> np.ndarray:
    """Generate evenly spaced weight vectors on the unit simplex.

    Uses the same approach as MOEA/D: place points on a lattice in the
    (num_objectives - 1)-simplex with `num_divisions` intervals per axis.

    Args:
        num_objectives: Number of objectives.
        num_divisions: Number of divisions along each axis.

    Returns:
        Array of shape (n_weights, num_objectives) with rows summing to 1.
    """
    combs = product(range(num_divisions + 1), repeat=num_objectives)
    weights = np.array([c for c in combs if sum(c) == num_divisions])
    weights = weights / num_divisions
    # Avoid zero weights — replace with small epsilon for Chebyshev stability
    weights = np.maximum(weights, 1e-4)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights


def build_neighbourhoods(weights: np.ndarray, neighbourhood_size: int) -> list:
    """Pre-compute the neighbourhood for each weight vector.

    For each weight vector w_i, finds the `neighbourhood_size` closest
    weight vectors by Euclidean distance (including w_i itself), exactly
    as MOEA/D does.

    Args:
        weights: Array of shape (W, num_objectives).
        neighbourhood_size: Number of neighbours per weight vector.

    Returns:
        List of length W, each element an array of neighbour indices.
    """
    neighbourhoods = []
    for w in weights:
        dists = np.linalg.norm(weights - w, axis=1)
        neighbours = np.argsort(dists)[:neighbourhood_size]
        neighbourhoods.append(neighbours)
    return neighbourhoods


class PQL(MOAgent):
    """Pareto Q-learning with three paradigm-specific scoring criteria.

    Implements a controlled comparison of three multi-objective selection
    paradigms on a shared Q-learning substrate:

    - ``'pareto'``       — Pareto cardinality scoring (original PQL).
    - ``'indicator'``    — Hypervolume indicator scoring (original PQL).
    - ``'decomposition'``— Chebyshev scalarisation over a neighbourhood of
                           weight vectors, analogous to MOEA/D selection.

    The pareto and indicator variants share identical data structures and
    Bellman updates: ``non_dominated[s][a]`` stores the full Pareto-pruned
    set of Q-vectors at each state-action pair.

    The decomposition variant uses a separate ``nd_decomp[s][a]`` structure
    that stores the best Q-vector per weight vector (via Chebyshev
    scalarisation) rather than the full non-dominated set. This gives the
    Bellman update the same directional pressure as MOEA/D's
    ``_update_solution``, ensuring the bootstrapped values actively guide
    exploration toward uncovered subproblems. Without this, the decomposition
    scorer would be applied at action selection only, while the update remains
    Pareto-based — causing systematic underexploration of regions not yet
    covered by the Pareto front.

    All three variants share: avg_reward update, ε-greedy exploration,
    episode structure, hyperparameters, and the get_local_pcs result
    extraction (which always applies final Pareto pruning for a fair
    comparison of the recovered front).

    Paper: K. Van Moffaert and A. Nowé, "Multi-objective reinforcement
    learning using sets of pareto dominating policies," JMLR, vol. 15,
    no. 1, pp. 3483–3512, 2014.
    """

    def __init__(
            self,
            env,
            ref_point,
            gamma=0.8,
            initial_epsilon=1.0,
            epsilon_decay_steps=100000,
            final_epsilon=0.1,
            seed=None,
            num_weight_divisions=5,
            neighbourhood_size=5,
            nd_update_freq=1
    ):
        """Initialise PQL.

        Args:
            env: The MO-Gymnasium environment.
            ref_point: Reference point for hypervolume calculation.
            gamma: Discount factor.
            initial_epsilon: Initial ε for ε-greedy exploration.
            epsilon_decay_steps: Steps over which ε decays linearly.
            final_epsilon: Final ε after decay.
            seed: Random seed.
            num_weight_divisions: Number of divisions for weight vector
                generation (used by the decomposition scorer). The total
                number of weight vectors W = C(num_objectives +
                num_weight_divisions - 1, num_weight_divisions).
            neighbourhood_size: Number of neighbouring weight vectors
                used by the decomposition scorer. Must be <= W.
                Smaller values reduce per-step cost; typical value is 5–10.
        """
        super().__init__(env, seed=seed)

        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.ref_point = ref_point

        # Action space
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

        # Observation space
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.env_shape = (self.env.observation_space.n,)
        elif isinstance(self.env.observation_space, gym.spaces.MultiDiscrete):
            self.env_shape = self.env.observation_space.nvec
        elif (
                isinstance(self.env.observation_space, gym.spaces.Box)
                and self.env.observation_space.is_bounded(manner="both")
                and issubclass(self.env.observation_space.dtype.type, numbers.Integral)
        ):
            low_bound = np.array(self.env.observation_space.low)
            high_bound = np.array(self.env.observation_space.high)
            self.env_shape = high_bound - low_bound + 1
        else:
            raise Exception("PQL only supports discretizable observation spaces.")

        self.num_states = np.prod(self.env_shape)
        self.num_objectives = self.env.unwrapped.reward_space.shape[0]

        # Q-learning data structures for pareto and indicator variants.
        # non_dominated[s][a]: full Pareto-pruned set of Q-vectors.
        self.counts = defaultdict(lambda: np.zeros(self.num_actions))

        self.non_dominated = defaultdict(
            lambda: [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
        )
        self.avg_reward = defaultdict(
            lambda: np.zeros((self.num_actions, self.num_objectives))
        )
        self.nd_decomp = defaultdict(
            lambda: [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)]
        )

        # Decomposition structures.
        self.weights = generate_weights(self.num_objectives, num_weight_divisions)
        self.neighbourhood_size = min(neighbourhood_size, len(self.weights))
        self.neighbourhoods = build_neighbourhoods(self.weights, self.neighbourhood_size)
        # Initialise to -inf so the first update always takes effect,
        # regardless of whether rewards are negative (as in the fruit tree).
        self.ideal_point = np.full(self.num_objectives, -np.inf)
        self.nd_update_freq = nd_update_freq
        self.archive = set()

    # ------------------------------------------------------------------
    # Ideal point
    # ------------------------------------------------------------------

    def _update_ideal_point_global(self, decomp: bool = False):
        """Update the ideal point globally from Q-sets across all visited
        state-action pairs.

        Reads Q-sets rather than avg_reward directly because Q-sets
        (avg_reward + gamma * bootstrap) reflect the true value estimate
        including discounted future rewards. For episodic environments with
        terminal rewards only (e.g. fruit tree), Q-sets equal avg_reward, so
        there is no difference. For non-terminal environments (e.g. lake),
        avg_reward captures only per-step reward and would systematically
        underestimate the ideal point, reducing the discriminative power of
        the Chebyshev scorer.

        Args:
            decomp: If True, read Q-sets from nd_decomp (decomposition
                    variant). If False, read from non_dominated
                    (pareto/indicator variants).

        Called once per episode end so the cost is amortised over the
        episode length rather than paid at every step.
        """
        for s in self.counts:
            for a in range(self.num_actions):
                if self.counts[s][a] == 0:
                    continue
                for vec in self.get_q_set(s, a, decomp=decomp):
                    self.ideal_point = np.maximum(self.ideal_point, np.array(vec))

    # ------------------------------------------------------------------
    # Scoring functions — THE ONLY DIFFERENCE between the three variants
    # ------------------------------------------------------------------

    def score_pareto_cardinality(self, state: int):
        """Pareto-based scoring: count non-dominated contributions per action."""
        visited = {a: self.get_q_set(state, a)
                   for a in range(self.num_actions)
                   if self.counts[state][a] > 0}
        if not visited:
            return np.ones(self.num_actions)  # uniform -> random selection
        candidates = set().union(*visited.values())
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)
        for vec in non_dominated:
            for a, q_set in visited.items():
                if vec in q_set:
                    scores[a] += 1
        return scores

    def score_hypervolume(self, state: int):
        """Indicator-based scoring: hypervolume contribution per action."""
        visited = [a for a in range(self.num_actions)
                   if self.counts[state][a] > 0]
        if not visited:
            return np.ones(self.num_actions)
        scores = np.zeros(self.num_actions)
        for a in visited:
            scores[a] = hypervolume(self.ref_point,
                                    list(self.get_q_set(state, a)))
        return scores

    def score_decomposition(self, state: int):
        """Decomposition-based scoring using Chebyshev scalarisation.

        Selects a random subproblem i, then evaluates only the k weight
        vectors in its neighbourhood B(i) — analogous to MOEA/D's
        neighbourhood-restricted update. The action that wins the most
        subproblems within the neighbourhood receives the highest score.

        Uses nd_decomp (not non_dominated) as the bootstrap structure, so
        the Q-sets reflect decomposition-directed exploration rather than
        Pareto pruning.

        Complexity: O(k × A × V) per call, where k = neighbourhood_size,
        A = num_actions, V = average Q-set size. This is independent of
        the total number of weight vectors W.
        """
        visited = {a: self.get_q_set(state, a, decomp=True)
                   for a in range(self.num_actions)
                   if self.counts[state][a] > 0}
        if not visited:
            return np.ones(self.num_actions)
        scores = np.zeros(self.num_actions)
        i = self.np_random.integers(len(self.weights))
        for idx in self.neighbourhoods[i]:
            w = self.weights[idx]
            best_action = None
            best_scalarised = np.inf
            for a, q_set in visited.items():
                for vec in q_set:
                    scalarised = np.max(w * (self.ideal_point - np.array(vec)))
                    if scalarised < best_scalarised:
                        best_scalarised = scalarised
                        best_action = a
            if best_action is not None:
                scores[best_action] += 1
        return scores

    # ------------------------------------------------------------------
    # Shared Q-learning machinery
    # ------------------------------------------------------------------

    def get_q_set(self, state: int, action: int, decomp: bool = False):
        """Compute the Q-set for a (state, action) pair.

        Args:
            state: Flattened state index.
            action: Action index.
            decomp: If True, bootstrap from nd_decomp (decomposition variant).
                    If False, bootstrap from non_dominated (pareto/indicator).
        """
        nd = self.nd_decomp[state][action] if decomp else self.non_dominated[state][action]
        nd_array = np.array(list(nd))
        q_array = self.avg_reward[state][action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def select_action(self, state: int, score_func: Callable):
        """ε-greedy action selection using the given scoring function."""
        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(
                np.argwhere(action_scores == np.max(action_scores)).flatten()
            )

    def calc_non_dominated(self, state: int):
        candidates = set().union(*[
            self.get_q_set(state, a)
            for a in range(self.num_actions)
            if self.counts[state][a] > 0
        ])
        if not candidates:
            return {tuple(np.zeros(self.num_objectives))}
        if len(candidates) == 1:
            return candidates
        return get_non_dominated(candidates)

    def calc_decomp_best(self, state: int) -> set:
        """Return the global best-per-weight Q-vector set at a state.

        For each weight vector w, finds the Q-vector across all actions
        that minimises the Chebyshev scalarisation. The resulting set —
        one vector per weight, duplicates collapsed — is the decomposition
        analog of calc_non_dominated: a single shared bootstrap for all
        actions at this state.
        """
        q_sets = [
            self.get_q_set(state, a, decomp=True)
            for a in range(self.num_actions)
            if self.counts[state][a] > 0  # exclude unvisited actions
        ]
        if not q_sets:
            return {tuple(np.zeros(self.num_objectives))}
        global_set = set()
        for w in self.weights:
            best_vec = None
            best_scalarised = np.inf
            for q_set in q_sets:
                for vec in q_set:
                    scalarised = np.max(w * (self.ideal_point - np.array(vec)))
                    if scalarised < best_scalarised:
                        best_scalarised = scalarised
                        best_vec = vec
            if best_vec is not None:
                global_set.add(best_vec)
        return global_set if global_set else {tuple(np.zeros(self.num_objectives))}

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
            self,
            total_timesteps: int,
            action_eval: str = "indicator",
            log_every: int = 1000,
    ):
        """Train the agent.

        Args:
            total_timesteps: Total environment steps to train for.
            action_eval: Scoring function to use. One of:
                - ``'pareto'``        Pareto cardinality scoring.
                - ``'indicator'``     Hypervolume indicator scoring.
                - ``'decomposition'`` Neighbourhood Chebyshev scoring.
            log_every: Log convergence metrics every this many steps.

        Returns:
            Tuple of (pareto_coverage_set, convergence_log).
        """
        if action_eval == "indicator":
            score_func = self.score_hypervolume
        elif action_eval == "pareto":
            score_func = self.score_pareto_cardinality
        elif action_eval == "decomposition":
            score_func = self.score_decomposition
        else:
            raise Exception(
                f"Unknown action_eval: '{action_eval}'. "
                f"Choose from 'pareto', 'indicator', 'decomposition'."
            )

        is_decomp = (action_eval == "decomposition")

        convergence_log = []
        next_log_step = log_every

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                action = self.select_action(state, score_func)
                if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                    action_nd = np.unravel_index(action, self.env.action_space.nvec)
                    next_state, reward, terminated, truncated, _ = self.env.step(action_nd)
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                # avg_reward update — identical for all three variants
                self.counts[state][action] += 1

                # Bellman update — paradigm-specific
                if self.global_step % self.nd_update_freq == 0:
                    if is_decomp:
                        self.nd_decomp[state][action] = self.calc_decomp_best(next_state)
                    else:
                        self.non_dominated[state][action] = self.calc_non_dominated(next_state)

                self.avg_reward[state][action] += (
                        (reward - self.avg_reward[state][action]) / self.counts[state][action]
                )

                state = next_state

                # Convergence logging
                if self.global_step >= next_log_step:
                    pcs = self.get_local_pcs(state=0, decomp=is_decomp)

                    # Exclude the zero-initialisation phantom vector — it dominates all
                    # real (negative) reward vectors under maximisation convention and
                    # would permanently corrupt the archive once added.
                    real_pcs = {v for v in pcs if any(x != 0.0 for x in v)}

                    self.archive |= real_pcs
                    if len(self.archive) > 1:
                        self.archive = get_non_dominated(self.archive)

                    hv = hypervolume(self.ref_point, list(self.archive)) if self.archive else 0.0
                    convergence_log.append({
                        "timestep": self.global_step,
                        "hypervolume": hv,
                        "pcs_size": len(self.archive),
                        "epsilon": self.epsilon,
                    })
                    next_log_step += log_every

            # Update the ideal point globally from all visited (s, a) pairs.
            # Done once per episode rather than per step to amortise cost.
            self._update_ideal_point_global(decomp=is_decomp)

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        # Final archive update before returning
        pcs = self.get_local_pcs(state=0, decomp=is_decomp)
        real_pcs = {v for v in pcs if any(x != 0.0 for x in v)}
        self.archive |= real_pcs
        if len(self.archive) > 1:
            self.archive = get_non_dominated(self.archive)
        return self.archive, convergence_log

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_local_pcs(self, state: int = 0, decomp: bool = False):
        """Return the Pareto coverage set at the given state.

        Always applies final Pareto pruning regardless of variant, so the
        returned set is directly comparable across all three paradigms.

        Args:
            state: Flattened state index.
            decomp: If True, build candidates from nd_decomp Q-sets.
        """
        q_sets = [
            self.get_q_set(state, a, decomp=decomp)
            for a in range(self.num_actions)
            if self.counts[state][a] > 0  # exclude unvisited actions
        ]
        if not q_sets:
            return set()
        candidates = set().union(*q_sets)
        if not candidates:
            return set()
        if len(candidates) == 1:
            return candidates
        return get_non_dominated(candidates)

    def get_config(self):
        return {
            "env_id": self.env.unwrapped.spec.id if hasattr(self.env.unwrapped, "spec") else "FruitTree",
            "gamma": self.gamma,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "final_epsilon": self.final_epsilon,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "num_objectives": self.num_objectives,
            "ref_point": self.ref_point.tolist(),
            "num_weights": len(self.weights),
            "neighbourhood_size": self.neighbourhood_size,
            "nd_update_freq": self.nd_update_freq,
        }
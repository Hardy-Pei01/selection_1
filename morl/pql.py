"""Pareto Q-Learning."""

import numbers
from itertools import product
from typing import Callable

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


class PQL(MOAgent):
    
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
    ):

        super().__init__(env, seed=seed)
        # Learning parameters
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon

        # Algorithm setup
        self.ref_point = ref_point

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.num_actions = self.env.action_space.n
        elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
            self.num_actions = np.prod(self.env.action_space.nvec)
        else:
            raise Exception("PQL only supports (multi)discrete action spaces.")

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
        self.counts = np.zeros((self.num_states, self.num_actions))
        self.non_dominated = [
            [{tuple(np.zeros(self.num_objectives))} for _ in range(self.num_actions)] for _ in range(self.num_states)
        ]
        self.avg_reward = np.zeros((self.num_states, self.num_actions, self.num_objectives))

        # Decomposition setup
        self.weights = generate_weights(self.num_objectives, num_weight_divisions)
        self.ideal_point = np.zeros(self.num_objectives)

    def _update_ideal_point(self, state: int):
        """Update the ideal point from the Q-sets of all actions at a state."""
        for action in range(self.num_actions):
            for vec in self.get_q_set(state, action):
                self.ideal_point = np.maximum(self.ideal_point, np.array(vec))

    def score_pareto_cardinality(self, state: int):

        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        non_dominated = get_non_dominated(candidates)
        scores = np.zeros(self.num_actions)

        for vec in non_dominated:
            for action, q_set in enumerate(q_sets):
                if vec in q_set:
                    scores[action] += 1

        return scores

    def score_hypervolume(self, state: int):

        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        action_scores = [hypervolume(self.ref_point, list(q_set)) for q_set in q_sets]
        return action_scores

    def score_decomposition(self, state: int):
        """Score actions using Chebyshev scalarisation over a set of weight vectors.

        For each weight vector, finds the best Q-vector across all actions
        using the Chebyshev (min-max) scalarisation. The action that
        contributes the best scalarised value for the most weight vectors
        receives the highest score — analogous to how MOEA/D selects
        solutions that best serve their assigned subproblems.
        """
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        scores = np.zeros(self.num_actions)

        for w in self.weights:
            best_action = None
            best_scalarised = np.inf

            for action, q_set in enumerate(q_sets):
                for vec in q_set:
                    # Chebyshev: minimise the worst weighted shortfall from ideal
                    scalarised = np.max(w * (self.ideal_point - np.array(vec)))
                    if scalarised < best_scalarised:
                        best_scalarised = scalarised
                        best_action = action

            if best_action is not None:
                scores[best_action] += 1

        return scores

    def get_q_set(self, state: int, action: int):

        nd_array = np.array(list(self.non_dominated[state][action]))
        q_array = self.avg_reward[state, action] + self.gamma * nd_array
        return {tuple(vec) for vec in q_array}

    def select_action(self, state: int, score_func: Callable):

        if self.np_random.uniform(0, 1) < self.epsilon:
            return self.np_random.integers(self.num_actions)
        else:
            action_scores = score_func(state)
            return self.np_random.choice(np.argwhere(action_scores == np.max(action_scores)).flatten())

    def calc_non_dominated(self, state: int):

        candidates = set().union(*[self.get_q_set(state, action) for action in range(self.num_actions)])
        non_dominated = get_non_dominated(candidates)
        return non_dominated

    def train(
            self,
            total_timesteps,
            action_eval="hypervolume",
            log_every=1000,
    ):

        if action_eval == "indicator":
            score_func = self.score_hypervolume
        elif action_eval == "pareto":
            score_func = self.score_pareto_cardinality
        elif action_eval == "decomposition":
            score_func = self.score_decomposition
        else:
            raise Exception(f"Unknown action_eval: {action_eval}. "
                            f"Choose from 'indicator', 'pareto', 'decomposition'.")

        convergence_log = []
        next_log_step = log_every

        while self.global_step < total_timesteps:
            state, _ = self.env.reset()
            state = int(np.ravel_multi_index(state, self.env_shape))
            terminated = False
            truncated = False

            while not (terminated or truncated) and self.global_step < total_timesteps:
                # Update ideal point before selection so decomposition scorer is current
                self._update_ideal_point(state)

                action = self.select_action(state, score_func)
                if isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                    action_nd = np.unravel_index(action, self.env.action_space.nvec)
                    next_state, reward, terminated, truncated, _ = self.env.step(action_nd)
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.global_step += 1
                next_state = int(np.ravel_multi_index(next_state, self.env_shape))

                self.counts[state, action] += 1
                self.non_dominated[state][action] = self.calc_non_dominated(next_state)
                self.avg_reward[state, action] += (reward - self.avg_reward[state, action]) / self.counts[state, action]

                state = next_state

                # Periodically log convergence metrics
                if self.global_step >= next_log_step:
                    pcs = self.get_local_pcs(state=0)
                    hv = hypervolume(self.ref_point, list(pcs)) if pcs else 0.0
                    convergence_log.append({
                        'timestep': self.global_step,
                        'hypervolume': hv,
                        'pcs_size': len(pcs),
                        'epsilon': self.epsilon,
                    })
                    next_log_step += log_every

            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                0,
                self.final_epsilon,
            )

        return self.get_local_pcs(state=0), convergence_log

    def get_local_pcs(self, state: int = 0):

        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        candidates = set().union(*q_sets)
        return get_non_dominated(candidates)
import numpy as np
import pandas as pd
import gymnasium as gym


def get_ind(pos):
    return int(2**pos[0] - 1) + pos[1]


class FruitTreeEnv(gym.Env):

    def __init__(self, depth, reward_dim, csv_path, observe,
                 scenario_index=None,
                 slip_patterns_path=None):
        super().__init__()
        self.reward_dim = reward_dim
        self.tree_depth = depth
        self.observe = observe

        branches = np.zeros((int(2 ** self.tree_depth - 1), self.reward_dim))
        df = pd.read_csv(csv_path)
        reward_cols = [c for c in df.columns if c.startswith('r')]
        fruits = df[reward_cols].values
        self.tree = np.concatenate([branches, fruits])

        self.reward_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=(self.reward_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=2 ** self.tree_depth - 1, shape=(2,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(2)
        self.current_state = np.array([0, 0])
        self.terminal = False

        # Generate slip pattern once at construction — the model is fixed
        # for the lifetime of this environment instance.
        if scenario_index is not None:
            patterns = np.load(slip_patterns_path)
            self._slip_pattern = patterns[int(scenario_index)]
        else:
            self._slip_pattern = None

    def get_tree_value(self, pos):
        return self.tree[get_ind(pos)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = np.array([0, 0])
        self.terminal = False
        return self.current_state.copy(), {}

    def step(self, action):
        if self._slip_pattern is not None:
            node_idx = get_ind(self.current_state)
            if self._slip_pattern[node_idx]:
                action = 1 - action
        # No stochastic fallback: slip_prob is only used via scenario_seed

        direction = {
            0: np.array([1, self.current_state[1]]),
            1: np.array([1, self.current_state[1] + 1]),
        }[action]

        self.current_state = self.current_state + direction
        reward = self.get_tree_value(self.current_state)

        if self.current_state[0] == self.tree_depth:
            self.terminal = True

        obs = (self.current_state.copy() if self.observe
               else np.array([0, self.current_state[1]],
                             dtype=np.int32))
        return obs, reward, self.terminal, False, {}

    @property
    def unwrapped(self):
        return self
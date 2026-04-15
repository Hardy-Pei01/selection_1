import numpy as np
import pandas as pd
from gymnasium import spaces


class FruitTreeEnv(object):

    def __init__(self, depth, reward_dim, csv_path, observe, slip_prob=0.0):

        self.reward_dim = reward_dim
        self.tree_depth = depth  # zero based depth
        self.observe = observe
        self.slip_prob = slip_prob
        self.rng = np.random.default_rng()

        branches = np.zeros((int(2**self.tree_depth-1), self.reward_dim))
        df = pd.read_csv(csv_path)
        reward_cols = [c for c in df.columns if c.startswith('r')]
        fruits = df[reward_cols].values
        self.tree = np.concatenate([branches, fruits])

        self.reward_space = spaces.Box(low=-10.0, high=0.0, shape=(self.reward_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=2**self.tree_depth-1, shape=(2,), dtype=np.int32)
        # action space specification: 0 left, 1 right
        self.action_space = spaces.Discrete(2)

        self.current_state = np.array([0, 0])
        self.terminal = False

    def get_ind(self, pos):
        return int(2 ** pos[0] - 1) + pos[1]

    def get_tree_value(self, pos):
        return self.tree[self.get_ind(pos)]

    def reset(self, seed=None):
        '''
            reset the location of the submarine
        '''
        self.current_state = np.array([0, 0])
        self.terminal = False
        self.rng = np.random.default_rng(seed)
        return self.current_state.copy()

    def step(self, action):
        '''
            step one move and feed back reward
        '''
        if self.rng.random() < self.slip_prob:
            action = 1 - action

        direction = {
            0: np.array([1, self.current_state[1]]),  # left
            1: np.array([1, self.current_state[1] + 1]),  # right
        }[action]

        self.current_state = self.current_state + direction

        reward = self.get_tree_value(self.current_state)
        if self.current_state[0] == self.tree_depth:
            self.terminal = True

        return self.current_state, reward, self.terminal
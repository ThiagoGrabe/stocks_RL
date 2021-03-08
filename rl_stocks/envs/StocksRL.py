# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

import numpy as np
import yfinance as yf
import random


class StocksRL(gym.Env):
    """
    Main class for stock trading gym environment. The main functions are: reset and step. Both should not be removed.

    param N_DISCRETE_ACTIONS: list of actions [0,1,2] > [buy, sell, hold]
    param OBS: observation space is the window interval to train. It indicates the input shape of the first layer of the NN.
    param action_space: Is also a gym space object that describes the action space, so the type of action that can be taken
    param observation_space: Is one of the gym spaces (Discrete, Box, ...) and describe the type and shape of the observation
    param data: The stock data
    param index: Simple counter for a dummy env.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(StocksRL, self).__init__()

        
    def _reset(self, actions, observation_space, data):
        self.N_DISCRETE_ACTIONS = actions
        self.OBS                = observation_space
        
        self.action_space       = spaces.Discrete(len(self.N_DISCRETE_ACTIONS))
        self.observation_space  = spaces.Box(low=0, high=float('inf'),
                                        shape=(self.OBS,), dtype=np.float32)
        self.data = data

        self.index = 0

    def reset(self):
        return self.perception()
    
    def step(self, action):
        self.obs    = self.perception()
        self.action = self._action(self.obs)
        self.reward = self._reward(self.action)
        self.done   = True
        self.info   = {}
        return self.obs, self.reward, self.done, self.info
    
    def render(self, mode='human'):
        print('render function')

    def perception(self):
        obs = np.random.choice(self.data, self.OBS)
        self.last_price = obs[-1]
        self.second_to_last = obs[-2]
        self.index += 1
        return obs

    def _action(self, obs):
        return random.choice(self.N_DISCRETE_ACTIONS)

    def _reward(self, action):
        if action == 0: # Buy
            if self.second_to_last > self.last_price:
                reward = self.last_price+self.second_to_last/2
            else:
                reward = self.last_price+self.second_to_last/2*(-1)
        elif action == 1: # Sell
            if self.second_to_last > self.last_price:
                reward = self.last_price+self.second_to_last/2*(-1)
            else:
                reward = self.last_price+self.second_to_last/2
        elif action == 2: # Hold
            if self.second_to_last > self.last_price:
                reward = self.last_price+self.second_to_last/2*(-1)
            else:
                reward = self.last_price+self.second_to_last/2

        return reward




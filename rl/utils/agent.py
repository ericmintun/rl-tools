'''
Helper classes and functions for agents.
'''

import numpy as np
from numpy.random import choice


class ReplayMemory():
    '''
    Stores state transitions, which consists of an observation, an
    action, the reward that action provided, and the observation of the
    new state.  Can return a random subset of the transitions.
    '''

    def __init__(self, length, obs_shape):
        self.length = length
        self.index = 0
        self.memory = np.zeros((self.length,) + obs_shape)
        self.is_full = False

    def add(self, obs):
        if obs.shape != self.obs_shape:
            raise ValueError("Observation shape does not match initialized 
                observation shape.")
        self.memory[self.index] = obs
        self.index += 1
        if self.index >= self.length:
            self.is_full = True
            self.index = 0

    def get_batch(self, batch_size):
        if is_full and batch_size < self.length:
            return self.memory[choice(self.length, batch_size, replace=False)]
        elif not is_full and batch_size < self.index:
            return self.memory[choice(self.index, batch_size, replace=False)]
        else:
            raise ValueError("Requested batch size larger than stored memory.")

    def reset(self):
        self.memory = np.zeros((self.length,) + obs_shape)
        self.is_full = False
        self.index = 0


            

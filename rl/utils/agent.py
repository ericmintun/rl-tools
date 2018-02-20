'''
Helper classes and functions for agents.
'''

import numpy as np
from numpy.random import choice

class Transition():
    '''
    A struct for storing transitions between states.  Stores the
    initial observation, action taken, reward received, final
    observation, and whether the final observation is a terminal state.

    This may either be used to describe single state transitions, or a
    batch of state transitions, if the dimension of every input is
    increased by one.  In the latter case, extra transitions may be
    appended.

    This thing is badly designed.
    '''
    def __init__(self, init_obs, action, reward, final_obs, done):
        self.init_obs = init_obs
        self.action = action
        self.reward = reward
        self.final_obs = final_obs
        self.done = done

    def append(self, transition):
        self.init_obs.append(transition.init_obs)
        self.action.append(transition.action)
        self.reward.append(transition.reward)
        self.final_obs.append(transition.final_obs)
        self.done.append(transition.done)

class ReplayMemory():
    '''
    Stores state transitions, which consists of an observation, an
    action, the reward that action provided, and the observation of the
    new state.  Can return a random subset of the transitions.  Returns
    such a subset as a single Transition instance with batched values.

    Initialization parameters:
    length (int) : the total number of state transitions to store.  Once
      the maximum is reached, newer transitions replace the oldest
      transitions

    Methods:
    add(transition) : adds the transition to memory.  'transition' must 
      be the Transition class holding a single transition.
    get_batch(batch_size) : returns 'batch_size' number of randomly
      selected state transitions as a single Transition class with
      batched inputs
    reset() : removes all stored state transitions.
    '''

    def __init__(self, length):
        self.length = length
        self.index = 0
        self.init_obs = []
        self.action = []
        self.reward = []
        self.final_obs = []
        self.done = []
        self.is_full = False

    def add(self, transition):
        if self.is_full:
            self.init_obs[self.index] = transition.init_obs
            self.action[self.index] = transition.action
            self.reward[self.index] = transition.reward
            self.final_obs[self.index] = transition.final_obs
            self.done[self.index] = transition.done
            self.index += 1
            if self.index >= self.length:
                self.index = 0
        else:
            self.init_obs.append(transition.init_obs)
            self.action.append(transition.action)
            self.reward.append(transition.reward)
            self.final_obs.append(transition.final_obs)
            self.done.append(transition.done)
            self.index += 1
            if self.index >= self.length:
                self.index = 0
                self.is_full = True

    def get_batch(self, batch_size):
        if is_full and batch_size < self.length:
            indices = choice(self.length, batch_size, replace=False)
        elif not is_full and batch_size < self.index:
            indices = choice(self.index, batch_size, replace=False)
        else:
            raise ValueError("Requested batch size larger than stored memory.")
        return Transition([self.init_obs[i] for i in indices],
                          [self.action[i] for i in indices],
                          [self.reward[i] for i in indices],
                          [self.final_obs[i] for i in indices],
                          [self.done[i] for i in indices])

    def reset(self):
        self.init_obs = []
        self.action = []
        self.reward = []
        self.final_obs = []
        self.done = []
        self.is_full = False
        self.index = 0


            

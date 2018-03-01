'''
Wrappers for OpenAI environments used by the examples.  Many of these
are quick implementations of wrappers found elsewhere.
'''

import gym
from gym import logger
import numpy as np

class ActionAtStart(gym.Wrapper):
    '''
    At the start of each new episode, take the specified action.
    '''
    def __init__(self, env, action_name='FIRE'):
        super(ActionAtStart, self).__init__(env)
        self.action = self.unwrapped.get_action_meanings().index(action_name)


    def reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(self.action) #Reset only gives observation
        return obs

class ActionRepeat(gym.Wrapper):
    '''
    Every time an action is performed, repeat it the specified number of
    times.  Only returns the observation data for the final output.  If
    the episode ends, stops repeating and returns the termination state.
    The reward is the sum of the rewards of the frames skipped.
    '''
    def __init__(self, env, repeat_number):
        super(ActionRepeat, self).__init__(env)
        self.repeat_number = repeat_number

    def step(self, action):
        total_reward = 0
        for i in range(self.repeat_number):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                return obs, total_reward, done, info
        return obs, total_reward, done, info

class EpisodeEndOnReward(gym.Wrapper):
    '''
    Terminates the episode as soon as any reward is recieved.
    '''
    def __init__(self, env):
        super(EpisodeEndOnReward,self).__init__(env)
        self.has_finished = False

    def step(self, action):
        if self.has_finished:
            logger.warn("EpisodeEndOnReward has ended the episode, but additional steps. have been taken.  Behavior may be undefined.")
        obs, reward, done, info = self.env.step(action)
        if reward != 0:
            self.has_finished = True
            return obs, reward, True, info
        return obs, 0, done, info

    def reset(self):
        self.has_finished = False
        return self.env.reset()

class RandomNoOpAtStart(gym.Wrapper):
    '''
    At the start of each episode, repeats the No Op action a random
    number of times, up to the specified maximum.  Uses the numpy random
    number generator.
    '''
    
    noop_name = 'NOOP'

    def __init__(self, env, max_actions):
        super(RandomNoOpAtStart, self).__init__(env)
        self.max_actions = max_actions
        self.noop = self.unwrapped.get_action_meanings().index(self.noop_name)

    def reset(self):
        obs = self.env.reset()
        num = np.random.choice(np.arange(self.max_actions))
        for i in range(num):
            obs, reward, done, info = self.env.step(self.noop)
            if done:
                logger.warn("Episode finished during initial noop actions.")
                return obs
        return obs


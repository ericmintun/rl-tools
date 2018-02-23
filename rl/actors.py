'''
Actors are in charge of actually choosing an action, given some input.

At the moment, these are just epsilon-greedy like functions, not full
actors as in the actor-critic approach.  I'm not sure this really makes
any sense but I need a place for it at the moment.
'''

import random

class DiscreteEpsilonGreedy:
    '''
    A simple epsilon-greedy algorithm for discrete action spaces.  Given 
    a best action to take, it takes that action with probability 
    (1 - epsilon) or takes a random action with probability epsilon.

    Initialization Arguments:
    action_space (int) : size of the action space to choose randomly
      from.  The act method returns a integer between 0 and
      (action_space - 1).
    epsilon (float or function) : the probability to choose a random
      action.  If a float should be between 0 and 1.  Can also be a
      single parameter function that takes in the number of training
      steps elapsed and returns a probability.

    Methods:
      act(best_action) : returns best_action with probability
        (1 - epsilon) or a random action otherwise.
      step() : increases the scheduling for the epsilon function by one
      reset() : resets the scheduling for the epsilon function to zero.
    '''

    def __init__(self, action_space, epsilon):
        self.action_space = action_space
        self.epsilon = epsilon
        self.count = 0

    def act(self, best_action):
        if callable(self.epsilon):
            p = self.epsilon(self.count)
        else:
            p = self.epsilon
        if random.random() < p:
            return random.choice(range(self.action_space))
        else:
            return best_action

    def step(self):
        self.count += 1

    def reset(self):
        self.count = 0

'''
Postprocessors are in charge of taking the output of a network and
producing definite actions, expected rewards, or other requested results
derived from the network output.

Postprocessors operate in torch variables since they need to connect
forward to the network trainer.
'''

import torch
from torch.autograd import Variable
import numpy as np

class DiscreteQPostprocessor:
    '''
    A postprocessor for Q learning in an environment with a discrete
    action space.  Assumes the network outputs an estimated value Q for
    each available action.  At the moment, this essentially does nothing
    but run the torch max function or pick values out of an array.

    Initialization values:
    none

    Methods:
    best_action(input, output_q=False) : 
      Input is a 2D array of estimated values Q of the form
      (batch, action).  Returns a 1D array of the actions with the 
      highest estimated rewards.  If output_q is True, returns a 2-tuple
      of the form (actions, q_values) where q_values are the values of
      the optimal actions.

    estimated_reward(input, actions) :
      Input is a 2D array of estimated values Q, and actions is a 1D
      array of actions of interest, which are integers between 0 and the
      length of the second dimension of input.  For each element in the
      batch, returns the ith Q given action i.
    '''

    def __init__(self):
        pass

    def best_action(self, input, output_q=False):
        
        q, action = torch.max(input, 1)

        if output_q == True:
            return (action, q)
        else:
            return action

    def estimated_reward(self, input, actions):
        print(type(actions))
        if type(actions) == Variable: #This isn't great
            return input[:,actions.data]
        else:
            return input[:,actions]


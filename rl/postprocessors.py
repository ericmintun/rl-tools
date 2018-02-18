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
    each available action.

    Initialization values:
    action_space (int) : The number of different allowable actions.
    use_gpu (bool) : Should be set to true if the output of the network
      is on the gpu.

    Methods:
    best_action(input, legal_actions=None, output_q=False, one_hot=True) : 
      Input is a 2D array of estimated values Q of the form
      (batch, action).  Returns the action with the highest estimated 
      reward.  If legal_actions is set to a list of integers of allowed 
      actions, the action from that list with the highest estimated 
      reward is used.

      Output form depends on flags set.  If one_hot is True, the 
      returned actions are a 2D array of the form (batch, action) where
      the max action is set to 1 and all other elements are 0.  If
      one_hot is False, returns a 1D array of integers where each
      integer is the label of the preferred action.  If output_q is
      True, return is a 2-tuple where the first element is as above,
      while the second is a 1D array of the Q associated with each
      output action.
    '''

    def __init__(self, action_space, use_gpu=False):
        self.action_space = action_space
        if use_gpu == True:
            self.dtype = torch.cuda.LongTensor
        else:
            self.dtype = torch.LongTensor

        self.label_mask = Variable(
                torch.from_numpy(
                    np.arange(self.action_space)).type(self.dtype),
                requires_grad=False)

    def best_action(self,
            input,
            legal_actions=None,
            output_q=False,
            one_hot=True)
        if legal_actions != None:
            if type(legal_actions) == self.dtype:
                indices = legal_actions
            elif type(legal_actions) == np.ndarray:
                indices = torch.from_numpy(legal_actions).type(self.dtype)
            else:
                indices = torch.from_numpy(
                        np.array(indices)).type(self.dtype)
            reduced = input[indices]
        else:
            reduced = input
        
        action, q = torch.max(reduced, 1)
        if one_hot==False:
            output = torch.sum(action * self.label_mask, dim=1)
        else:
            output = action

        if output_q == True:
            return (output, q)
        else:
            return output

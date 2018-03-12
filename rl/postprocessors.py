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
        if type(actions) == Variable: #This isn't great
            return torch.gather(input,1,actions.view(-1,1)).view(-1)
        else:
            return torch.gather(input,1,Variable(actions.view(-1,1))).view(-1)


class CapsuleBasicPostprocessor:
    '''
    A postprocessor designed for basic capsules.  Extracts probability of
    an entities existence from the length of the supplied pose vector.

    Initialization values:
    none

    Methods:
    predictions(input) :
        Input is a 3D tensor of the form (batch, label, pose_element).
        Returns a 2D tensor of the form (batch, label) where each
        element is a number from 0 to 1 yielding the predicted
        probability that element exists.

    mask(input, mask_vectors) :
        input is a 3D tensor of the form (batch, label, pose_element).
        mask_vector is a 2D tensor of the form (batch, label), where
        every element is a zero or a one.  Returns a 3D tensor of the
        same form as input, where every element of the pose vector
        corresponding to a zero in mask_vectors is set to zero.
    '''

    def __init__(self):
        pass

    def predictions(input):
        return torch.norm(input,dim=2)

    def mask(input, mask_vectors):
        if type(mask_vectors) is Variable:
            m = mask_vectors
        elif type(mask_vectors) is torch.Tensor:
            m = Variable(mask_vectors)
        else:
            raise TypeError("mask_vectors must be either a torch tensor or torch variable.")

        #Permute the pose_elements to the first index so multiply broadcasts correctly
        return (input.permute(2,0,1) * mask_vectors.type(torch.LongTensor)).permute(0,1,2)


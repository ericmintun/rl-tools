'''
Utility functions for working with torch objects.
'''

import torch
from torch.autograd import Variable
import numpy as np

def torchify(input_array, torch_type, requires_grad=False):
    '''
    Turns the input into a torch variable of the desired type.  Can take
    in a numpy array, torch tensor, or torch variable.  This does not
    preserve the integrity of gradient backpropagation if passed a 
    Variable, and it should not be used in the middle of systems that
    need backpropagation.  

    Args:
    input_array (numpy array, torch tensor, or torch variable) : 
      The data to put in a torch format
    torch_type (torch tensor class) : which type of torch tensor to cast
      the data to.
    requires_grad (bool) : whether or not the torch variable requires
      backpropagation
    '''
    if type(input_array) == np.ndarray:
        return Variable(torch.from_numpy(input_array).type(torch_type),
                requires_grad=requires_grad)
    elif type(input_array) == torch.Tensor:
        return Variable(input_array.type(torch_type), 
                requires_grad=requires_grad)
    elif type(input_array) == Variable:
        return Variable(input_array.data.type(torch_type), 
                requires_grad=requires_grad)
    else:
        raise TypeError("The only supported types are numpy arrays, torch tensors, and torch variables")
                


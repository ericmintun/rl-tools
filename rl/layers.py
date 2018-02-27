'''
Custom neural network layers.
'''

import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable

class RLConv2d(nn.Conv2d):
    '''
    This is a regular PyTorch Conv2d layer with a single extra feature: it can save a separate copy of 
    the weights and biases as a snapshot of the current weights and biases.  The feed forward step then 
    can be run with the actual weights or the weight snapshots.  The snapshots are not parameters of the 
    model and don't receive gradient update.  This layer is useful in double deep Q learning or anytime
    you want a slowly updating version of the network for stability reasons.

    Args:
        snapshot (bool, optional): Whether or not to initialize snapshot weights
        all args from torch.nn.Conv2d

    Attributes:
        weight_snapshot (Tensor): the snapshot of the weights.  Same shape as weight
        bias_snapshot (Tensor): the snapshot of the biases.  Same shape as bias
        all attributes of torch.nn.Conv2d
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=True, snapshot=True):

        super(RLConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias)

        self.snapshot = snapshot
        
        if snapshot == True:
            #self.weight_snapshot = Variable(self.weight.data, requires_grad=False)
            self.register_buffer('weight_snapshot',self.weight.data)
            if bias == True:
                #self.bias_snapshot = Variable(self.bias.data, requires_grad=False)
                self.register_buffer('bias_snapshot', self.bias.data)
            else:
                self.bias_snapshot = None
        else:
            self.weight_snapshot = None
            self.bias_snapshot = None

    def forward(self, input, use_snapshot=False):
        
        if use_snapshot == True:
            if self.snapshot == True:
                #return nnf.conv2d(input, self.weight_snapshot, self.bias_snapshot, self.stride,
                #    self.padding, self.dilation, self.groups)
                return nnf.conv2d(input, Variable(self._buffers['weight_snapshot']), Variable(self._buffers['bias_snapshot']), self.stride,
                    self.padding, self.dilation, self.groups)
                                
            else:
                raise ValueError("A feed forward step with fixed weights was requested from a layer without fixed weights initialized.")
        else:
            return nnf.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                    self.dilation, self.groups)


    def update_snapshot(self):
        if self.snapshot == True:
            self.weight_snapshot.data = self.weight.data
            self.bias_snapshot.data = self.bias.data
        else:
            raise ValueError("An update to fixed weights was requested from a layer without fixed weights initialized.")
                


class RLLinear(nn.Linear):
    '''
    This is a regular PyTorch Linear layer with a single extra feature: it can save a separate copy of 
    the weights and biases as a snapshot of the current weights and biases.  The feed forward step then 
    can be run with the actual weights or the weight snapshots.  The snapshots are not parameters of the 
    model and don't receive gradient update.  This layer is useful in double deep Q learning or anytime
    you want a slowly updating version of the network for stability reasons.    

    Args:
        snapshot (bool, optional): Whether or not to initialize snapshot weights
        all args from torch.nn.Linear

    Attributes:
        weight_snapshot (Tensor): the snapshot of the weights.  Same shape as weight
        bias_snapshot (Tensor): the snapshot of the biases.  Same shape as bias
        all attributes of torch.nn.Linear
    '''

    def __init__(self, in_features, out_features, bias=True, snapshot=True):
        self.snapshot = snapshot
        super(RLLinear, self).__init__(in_features, out_features, bias)
        
        if snapshot == True:
            self.weight_snapshot = Variable(self.weight.data, requires_grad=False)
            if bias == True:
                self.bias_snapshot = Variable(self.bias.data, requires_grad=False)
            else:
                self.bias_snapshot = None
        else:
            self.weight_snapshot = None
            self.bias_snapshot = None

    def forward(self, input, use_snapshot=False):
        
        if use_snapshot == True:
            if self.snapshot == True:
                return nnf.linear(input, self.weight_snapshot, self.bias_snapshot)
            else:
                raise ValueError("A feed forward step with fixed weights was requested from a layer without fixed weights initialized.")
        else:
            return nnf.linear(input, self.weight, self.bias)    

    def update_snapshot(self):
        if self.snapshot == True:
            self.weight_snapshot.data = self.weight.data
            self.bias_snapshot.data = self.bias.data
        else:
            raise ValueError("An update to fixed weights was requested from a layer without fixed weights initialized.")



class RLDropout(nn.Dropout):
    '''
    A version of dropout that allows for explicitly overriding the 'training' boolean set by the module.  This allows some forward
    passes to be run in non-training mode during training.  Potentially useful in RL since the forward pass is used both as the
    for determining both the target and prediction, but we are only training the prediction.

    Extra arg in forward:
    training_override (bool or None, optional): If set to None, use the module 'training' boolean, otherwise use this.
    '''

    def forward(self, input, training_override = None):
        if training_override == None:
            return nnf.dropout(input, self.p, self.training, self.inplace)
        else:
            return nnf.dropout(input, self.p, training_override, self.inplace)

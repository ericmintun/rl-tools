'''
Custom neural network layers.
'''

import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable, Parameter

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

    def forward(self, input, use_snapshot=False):
        
        if use_snapshot == True:
            if self.snapshot == True:
                w = Variable(self._buffers['weight_snapshot'])
                if self.bias is not None:
                    b = Variable(self._buffers['bias_snapshot'])
                else:
                    b = None
            else:
                raise ValueError("A feed forward step with fixed weights was requested from a layer without fixed weights initialized.")
        else:
            w = self.weight
            b = self.bias

        return nnf.conv2d(input, w, b, self.stride, self.padding,
                            self.dilation, self.groups)


    def update_snapshot(self):
        if self.snapshot == True:
            self._buffers['weight_snapshot'] = self.weight.data
            self._buffers['bias_snapshot'] = self.bias.data
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
            self.register_buffer('weight_snapshot', self.weight.data)
            if bias == True:
                self.register_buffer('bias_snapshot', self.bias.data)

    def forward(self, input, use_snapshot=False):
        
        if use_snapshot == True:
            if self.snapshot == True:
                w = Variable(self._buffers['weight_snapshot'])
                if self.bias is not None:
                    b = Variable(self._buffers['bias_snapshot'])
                else:
                    b = None
            else:
                raise ValueError("A feed forward step with fixed weights was requested from a layer without fixed weights initialized.")
        else:
            w = self.weight
            b = self.bias

        return nnf.linear(input, w, b)    

    def update_snapshot(self):
        if self.snapshot == True:
            self._buffers['weight_snapshot'] = self.weight.data
            self._buffers['bias_snapshot'] = self.bias.data
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


class RLCapsuleBasic(nn.Module):
    

    def __init__(self, in_channels, out_channels, in_pose_size, 
                  out_pose_size, bias=True, snapshot=True):
        super(RLCapsuleBasic, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_pose_size = in_pose_size
        self.out_pose_size = out_pose_size
        self.snapshot = snapshot
        self.u = Parameter(torch.Tensor(self.out_channels, self.in_channels,
                            self.out_pose_size, self.in_pose_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels, 
                                                self.in_channels,
                                                self.out_pose_size))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

        if snapshot == True:
            self.register_buffer('weight_snapshot', self.weight.data)
            if bias == True:
                self.register_buffer('bias_snapshot', self.bias.data)

    def reset_parameters(self):
        n = self.in_channels * (self.pose_size ** 2)
        stdv = 1 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, use_snapshot=False):

        #input is (batch, in_channel, vector_element)
        #uij is (out_channel, in_channel, y, x)
        #also a bias
        #uhatij is (batch, out_channel, in_channel, vec_hat_element)
        #   matched over in_channel, mat-mult uij.input in last dims
        #sj is (batch, out_channel, vector_element)
        #   sj = sum_i cij uhatij
        #   cij is (batch, out_channel, in_channel)
        #vj = ( sj / ||sj|| ) * ( ||sj||^2 / ( 1 + ||sj||^2 ) )

        #Routing loop
        #init bij = 1
        #loop:
        #cij = exp(-bij) / sum_i( exp(-bij) )
        #get vj from cj, uhatij as above
        #bij = bij + vj * uhatij

        batch_size = input.size[0]

        if use_snapshot == True:
            if self.snapshot == True:
                u = Variable(self._buffers['weight_snapshot'])
                if self.bias is not None:
                    bias = Variable(self._buffers['bias_snapshot'])
                else:
                    bias = None
            else:
                raise ValueError("A feed forward step with fixed weights was requested from a layer without fixed weights initialized.")
        else:
            u = self.u
            bias = self.bias

        u_hat = torch.matmul(u, input.unsqueeze(2))
        if self.bias is not None:
            u_hat = u_hat + bias

        b = torch.ones(batch_size, self.out_channels, self.in_channels)
        for i in range(self.routing_iters):
            c = nnf.softmax(b, 1)
            #Since torch doesn't support batch dot products and only supports
            #batch matrix multiplication in the last two indices, the following
            #mess is needed.
            #maps c to (batch, out_channel, 1, 1, in_channel)
            #maps u_hat to (batch, out_channel, vector_element, in_channel, 1)
            #matmul broadcasts to (batch, out_channel, vector_element, 1, 1)
            #squeeze reduces to desired (batch, out_channel, vector_element)
            s = torch.matmul(c.unsqueeze(2).unsqueeze(2), 
                    u_hat.permute(0,1,3,2).unsqueeze(4)).squeeze()
            v = s * (1 + s.norm(dim=2)**2)/ s.norm(dim=2)
            #Same problem, at least there is no permuting this time
            #Maps v to (batch, out_channel, 1, 1, vector_element)
            #Maps u_hat to (batch, out_channel, in_channel, vector_element, 1)
            #matmul gives (batch, out_channel, in_channel, 1, 1)
            #squeeze reduces to desires (batch, out_channel, in_channel)
            b = b + torch.matmul(v.unsqueeze(1).unsqueeze(3),
                    u_hat.unsqueeze(4)).squeeze()

        return v

        def update_snapshot(self):
            
            if self.snapshot == True:
                self._buffers['weight_snapshot'] = self.u.data
                self._buffers['bias_snapshot'] = self.bias.data
            else:
                raise ValueError("An update to fixed weights was requested from a layer without fixed weights initialized.")

        

'''
A basic CNN using RL layers.  The graph is two 2d convolutional layers
with stride 1 and dilation 1, two 2d pooling layers, and one dense fully
connected layer.  ReLU activations are on all steps.  The default 
setting are a typical MNIST net (Google's MNIST tutorial, specifically.
Input shape is a 3-tuple of the form (channels, y_size, x_size), output
shape is any tuple.

Only accepts 1D output_shapes
'''

import rl.layers as l
import torch.nn as nn

class RL_Capsule(nn.Module):

    def __init__(self, 
        input_shape = (1,28,28),
        output_shape = (10,),
        kernel_size1 = 9,
        kernel_size2 = 9,
        conv1_features = 256,
        conv1_stride = 1,
        conv2_stride = 2,
        caps1_features = 64,
        pose_size1 = 8,
        pose_size2 = 16):

        super(RL_Capsule, self).__init__()

        self.input_shape = input_shape
        self.output_length = output_shape[0]
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.conv1_features = conv1_features
        self.conv1_stride = conv1_stride
        self.conv2_stride = conv2_stride
        self.caps1_features = caps1_features
        self.pose_size1 = pose_size1
        self.pose_size2 = pose_size2

        self.conv2_features = self.caps1_features * self.pose_size1


        self.after_conv1_shape = ((self.input_shape[1] - self.kernel_size1) // self.conv1_stride + 1,
                                  (self.input_shape[2] - self.kernel_size1) // self.conv1_stride + 1)
        self.after_conv2_shape = ((self.after_conv1_shape[0] - self.kernel_size2) // self.conv2_stride + 1,
                                  (self.after_conv1_shape[1] - self.kernel_size2) // self.conv2_stride + 1)
        self.caps1_size = self.after_conv2_shape[0] \
                          * self.after_conv2_shape[1] \
                          * self.caps1_features


        self.padding = 0

        #Convolutional Layers
        self.conv1 = l.RLConv2d(self.input_shape[0], self.conv1_features, self.kernel_size1, padding=self.padding, stride=self.conv1_stride)
        self.conv2 = l.RLConv2d(self.conv1_features, self.conv2_features, self.kernel_size2, padding=self.padding, stride=self.conv2_stride)

        #Linear Layers
        self.caps = l.RLCapsuleBasic(self.caps1_size, self.output_length,
                                              self.pose_size1, self.pose_size2)

        #Activation
        self.relu = nn.ReLU()

    
    def forward(self, input, use_snapshot = False, training_override = None):
        batch_size = input.size()[0]
        layer1 = self.relu.forward(self.conv1.forward(input, use_snapshot))
        layer2 = self.conv2.forward(layer1, use_snapshot)
        layer2Reshape = layer2.permute(0,2,3,1).contiguous().view(-1,self.caps1_size,self.pose_size1)
        output = self.caps.forward(layer2Reshape, use_snapshot)

        return output

    def update_snapshot(self):
        self.conv1.update_snapshot()
        self.conv2.update_snapshot()
        self.caps.update_snapshot()
        return

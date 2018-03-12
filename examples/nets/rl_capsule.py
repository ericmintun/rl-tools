'''
A basic CNN using RL layers.  The graph is two 2d convolutional layers
with stride 1 and dilation 1, two 2d pooling layers, and one dense fully
connected layer.  ReLU activations are on all steps.  The default 
setting are a typical MNIST net (Google's MNIST tutorial, specifically.
Input shape is a 3-tuple of the form (channels, y_size, x_size), output
shape is any tuple.
'''

import rl.layers as l
import torch.nn as nn

class RL_Capsule(nn.Module):

    def __init__(self, 
        input_shape = (1,28,28),
        output_shape = (10,),
        kernel_size = 9,
        conv1_features = 256,
                  conv1_stride = 1,
                conv2_stride = 2,
        caps1_features = 64,
                pose_size1 = 8,
                pose_size2 = 16):

       super(RL_Capsule, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.conv1_features = conv1_features
        self.conv1_stride = conv1_stride
        self.conv2_stride = conv2_stride
        self.caps1_features = caps1_features
        self.pose_size1 = pose_size1
        self.pose_size2 = pose_size2

        self.conv2_features = self.caps1_features * self.pose_size1
        self.after_conv_shape = ( self.input_shape[[1]] // self.conv1_stride // self.conv2_stride,
                                  self.input_shape[[2]] // self.conv1_stride // self.conv2_stride )
        self.caps1_size = self.after_conv_shape[[0]] \
                          * self.after_conv_shape[[1]] \
                          * self.caps1_features


        self.padding = self.kernel_size // 2

        #Convolutional Layers
        self.conv1 = l.RLConv2d(self.input_shape[0], self.conv1_features, self.kernel_size, padding=self.padding, stride=self.conv1_stride)
        self.conv2 = l.RLConv2d(self.conv1_features, self.conv2_features, self.kernel_size, padding=self.padding, stride=self.conv2_stride)

        #Linear Layers
        self.caps = l.RLCapsuleBasic(self.caps1_features, self.output_shape,
                                              self.pose_size1, self.pose_size2)

        #Activation
        self.relu = nn.ReLU()

    
      def forward(self, input, use_snapshot = False, training_override = None):
        batch_size = input.size()[0]
        layer1 = self.relu.forward(self.conv1.forward(input, use_snapshot))
        layer2 = self.conv2.forward(layer1, use_snapshot)
        layer2Reshape = layer2.permute(0,2,3,1).view(-1,self.caps1_size,self.pos_size1)
        output = self.caps.forward(input, use_snapshot)

        return output

     def update_snapshot(self):
        self.conv1.update_snapshot()
        self.conv2.update_snapshot()
        self.caps.update_snapshot()
        return

'''
A basic CNN using RL layers.  The graph is two 2d convolutional layers
with stride 1 and dilation 1, two 2d pooling layers, and one dense fully
connected layer.  ReLU activations are on all steps.  The default 
setting are a typical MNIST net (Google's MNIST tutorial, specifically.
Input shape is a 3-tuple of the form (channels, y_size, x_size), output
shape is any tuple.
'''

import layers as l
import torch.nn as nn

class RL_CNN(nn.Module):

	def __init__(self, 
		input_shape = (1,28,28),
		output_shape = (10,),
		kernel_size = 5,
		conv1_features = 32,
		conv2_features = 64,
		lin1_features = 1024,
		pool_kernel_size = 2,
		dropout_p = 0.5):


		super(RL_CNN, self).__init__()

		self.input_shape = input_shape
		self.output_shape = output_shape
		self.kernel_size = kernel_size
		self.conv1_features = conv1_features
		self.conv2_features = conv2_features
		self.pool_kernel_size = pool_kernel_size
		self.lin1_features = lin1_features
		self.dropout_p = dropout_p
		self.padding = self.kernel_size // 2

		#Convolutional Layers
		self.conv1 = l.RLConv2d(self.input_shape[0], self.conv1_features, self.kernel_size, padding=self.padding)
		self.conv2 = l.RLConv2d(self.conv1_features, self.conv2_features, self.kernel_size, padding=self.padding)

		#Linear Layers
		self.lin1_in_channels = (self.input_shape[1] // (self.pool_kernel_size ** 2)) * (self.input_shape[2] // (self.pool_kernel_size ** 2)) * self.conv2_features
		self.lin1 = l.RLLinear(self.lin1_in_channels, self.lin1_features)
		
		self.lin2_out_channels = 1
		for n in self.output_shape:
			self.lin2_out_channels *= n
		self.lin2 = l.RLLinear(self.lin1_features, self.lin2_out_channels)

		#Activation
		self.relu = nn.ReLU()

		#Pooling
		self.pool = nn.MaxPool2d(self.pool_kernel_size)

		#Drop out
		self.dropout = l.RLDropout(self.dropout_p)

	
	def forward(self, input, use_snapshot = False, training_override = None):
		batch_size = input.size()[0]
		layer1 = self.relu.forward(self.pool.forward(self.conv1.forward(input, use_snapshot)))
		layer2 = self.relu.forward(self.pool.forward(self.conv2.forward(layer1, use_snapshot)))
		layer2Reshape = layer2.view((batch_size, self.lin1_in_channels))
		layer3 = self.dropout.forward(self.relu.forward(self.lin1(layer2Reshape, use_snapshot)), training_override)
		layer4 = self.lin2(layer3, use_snapshot)
		output = layer4.view((batch_size,) + self.output_shape)

		return output

	def update_snapshot(self):
		self.conv1.update_snapshot()
		self.conv2.update_snapshot()
		self.lin1.update_snapshot()
		self.lin2.update_snapshot()
		return

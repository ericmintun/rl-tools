'''
A fixed behavior convolutional neural network.  Parameters are hard-coded.  Can:
1. Take an input and yield an output
2. Update weights using gradient descent.  The gradients must be already accumulated in the weights via autograd being called farther down the PyTorch graph.
3. Take a snapshot of current weights.  This snapshot can be used instead of the up-to-date weights for feed forward.  They can't be backpropagated on.

Initializes with the input and output shapes, in the form (channels, xLength, yLength).

Current architecture:




'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from functools import reduce
from operator import mul as multiply

class CNNFixed:

	dropoutP = 0.4

	l1Kernel = 5
	l1Stride = 1
	l1Features = 32
	l1Pool = 2
	
	l2Kernel = 5
	l2Stride = 1
	l2Features = 64
	l2Pool = 2
	
	l3Features = 1024

	def __init__(self, inputShape, outputShape, qGPU=False):
		self._inputShape = inputShape
		self._outputShape = outputShape

		if qGPU == True:
			dtype = torch.cuda.FloatTensor
		else:
			dtype = torch.FloatTensor

		#Initialize Layer 0-1
		l1OutputShape = (self.l1Features, int(self._inputShape[1] / (self.l1Stride * self.l1Pool)), int(self._inputShape[2] / (self.l1Stride * self.l1Pool)))
		l1KernelShape = (l1OutputShape[0], self._inputShape[0], self.l1Kernel, self.l1Kernel)

		self._w1 = Variable(torch.rand(l1KernelShape).type(dtype)*0.627-0.627/2, requires_grad=True)
		self._w1Fixed = Variable(self._w1.data, requires_grad=False)

		self._b1 = Variable(torch.zeros(l1OutputShape[0]).type(dtype), requires_grad=True)
		self._b1Fixed = Variable(self._b1.data, requires_grad=False)

		#Initialize Layer 1-2
		l2OutputShape = (self.l2Features, int(l1OutputShape[1] / (self.l2Stride * self.l2Pool)), int(l1OutputShape[2] / (self.l2Stride * self.l2Pool)))
		l2KernelShape = (l2OutputShape[0], l1OutputShape[0], self.l2Kernel, self.l2Kernel)

		self._w2 = Variable(torch.rand(l2KernelShape).type(dtype)*0.167-0.167/2, requires_grad=True)
		self._w2Fixed = Variable(self._w2.data, requires_grad=False)

		self._b2 = Variable(torch.zeros(l2OutputShape[0]).type(dtype), requires_grad=True)
		self._b2Fixed = Variable(self._b2.data, requires_grad=False)

		#Initialize Layer 2-3
		self.l3InputSize = reduce(multiply, l2OutputShape)
		l3Shape = (self.l3InputSize, self.l3Features) 

		self._w3 = Variable(torch.rand(l3Shape).type(dtype)*0.0760-0.0760/2, requires_grad=True)
		self._w3Fixed = Variable(self._w3.data, requires_grad=False)

		self._b3 = Variable(torch.zeros(self.l3Features).type(dtype), requires_grad=True)
		self._b3Fixed = Variable(self._b3.data, requires_grad=False)
	
		#Initialize Layer 3-Out
		self.l4OutputSize = reduce(multiply, self._outputShape, 1)
		l4Shape = (self.l3Features, self.l4OutputSize)

		self._w4 = Variable(torch.rand(l4Shape).type(dtype)*0.152-0.152/2, requires_grad=True)
		self._w4Fixed = Variable(self._w4.data, requires_grad=False)

		self._b4 = Variable(torch.zeros(self.l4OutputSize).type(dtype), requires_grad=True)
		self._b4Fixed = Variable(self._b4.data, requires_grad=False)
		
		

	def shapshotWeights(self):
		self._w1Fixed.data = self._w1.data
		self._b1Fixed.data = self._b1.data
		self._w2Fixed.data = self._w2.data
		self._b2Fixed.data = self._b2.data
		self._w3Fixed.data = self._w3.data
		self._b3Fixed.data = self._b3.data
		self._w4Fixed.data = self._w4.data
		self._b4Fixed.data = self._b4.data
		return

	def feedForward(self, data, qDropout=False, qSnapshot=False):

		if qSnapshot == True:
			w1 = self._w1Fixed
			b1 = self._b1Fixed
			w1 = self._w2Fixed
			b1 = self._b2Fixed
			w1 = self._w3Fixed
			b1 = self._b3Fixed
			w1 = self._w4Fixed
			b1 = self._b4Fixed
		else:
			w1 = self._w1
			b1 = self._b1
			w2 = self._w2
			b2 = self._b2
			w3 = self._w3
			b3 = self._b3
			w4 = self._w4
			b4 = self._b4

		l1Padding = int(self.l1Kernel / 2)
		l1 = nnf.relu(nnf.max_pool2d(nnf.conv2d(data, w1, bias=b1, stride=self.l1Stride, padding=l1Padding),self.l1Pool))

		l2Padding = int(self.l2Kernel / 2)
		l2 = nnf.relu(nnf.max_pool2d(nnf.conv2d(l1, w2, bias=b2, stride=self.l2Stride, padding=l2Padding),self.l2Pool))
		
		batchSize = data.size()[0]
		l3Shape = (batchSize, self.l3InputSize)
		#l3 = nnf.dropout(nnf.relu(nnf.max_pool2d(nnf.conv2d(l2, w3, bias=b3, stride=self.l3Stride, padding=l3Padding),self.l3Pool)), self.dropoutP, qDropout)
		l3 = nnf.dropout(nnf.relu(l2.view(l3Shape).mm(w3)+b3),self.dropoutP, qDropout)
		output = l3.mm(w4)+b4

		outputShape = [batchSize]
		outputShape.extend([s for s in self._outputShape])
		return output.view(outputShape)

	def updateWeights(self, stepSize):
		self._w1.data -= stepSize * self._w1.grad.data
		self._b1.data -= stepSize * self._b1.grad.data
		self._w2.data -= stepSize * self._w2.grad.data
		self._b2.data -= stepSize * self._b2.grad.data
		self._w3.data -= stepSize * self._w3.grad.data
		self._b3.data -= stepSize * self._b3.grad.data
		self._w4.data -= stepSize * self._w4.grad.data
		self._b4.data -= stepSize * self._b4.grad.data

		self._w1.grad.data.zero_()
		self._b1.grad.data.zero_()
		self._w2.grad.data.zero_()
		self._b2.grad.data.zero_()
		self._w3.grad.data.zero_()
		self._b3.grad.data.zero_()
		self._w4.grad.data.zero_()
		self._b4.grad.data.zero_()

		return



if __name__ == "__main__":

	use_gpu = torch.cuda.is_available()

	
	testInputSize = (3,40,40)
	testOutputSize = (10,)
	testBatch = 50

	test = CNNFixed(testInputSize, testOutputSize, use_gpu)

	testInput = Variable(torch.randn(testBatch, testInputSize[0], testInputSize[1], testInputSize[2]).type(torch.cuda.LongTensor),requires_grad=False)
	output = test.feedForward(testInput, True)
	testOutput = Variable(torch.from_numpy(np.random.random_integers(0,9,size=(50))).type(torch.cuda.LongTensor), requires_grad=False)
	loss = nnf.cross_entropy(output, testOutput)
	loss.backward()
	test.updateWeights(1e-5)

	



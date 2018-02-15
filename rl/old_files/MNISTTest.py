'''
A neural net that can't at least partially learn MNIST is probably broken.  This provides a first line of defense for debugging the things.
'''

import numpy as np
import idx
from random import shuffle
import torch
import torch.nn as nn
from torch.autograd import Variable
import CNNFixed
import time

def getMNISTData(imagesFilename, labelsFilename):
	images = idx.read(imagesFilename)
	labels = idx.read(labelsFilename)
	if len(images) != len(labels):
		raise ValueError("The number of labels and number of images do not match.")
	return list(zip(images, labels))

class MNISTNet:

	imageX = 28
	imageY = 28
	channels = 1
	classes = 10
	trainImagesFilename = 'train-images.idx3-ubyte.gz'
	trainLabelsFilename = 'train-labels.idx1-ubyte.gz'
	testImagesFilename = 't10k-images.idx3-ubyte.gz'
	testLabelsFilename = 't10k-labels.idx1-ubyte.gz'

	def __init__(self, NN, batchSize=50, stepSize=1e-5, MNISTDirectory='./', qGPU=False, preProcessor=None):
		self.NN = NN
		self.batchSize = batchSize
		self.step = stepSize

		if qGPU == False:
			self.floatType = torch.FloatTensor
			self.longType = torch.LongTensor
		else:
			self.floatType = torch.cuda.FloatTensor
			self.longType = torch.cuda.LongTensor

		self.trainImagesPath = MNISTDirectory + self.trainImagesFilename
		self.trainLabelsPath = MNISTDirectory + self.trainLabelsFilename
		self.testImagesPath = MNISTDirectory + self.testImagesFilename
		self.testLabelsPath = MNISTDirectory + self.testLabelsFilename

		self.trainData = getMNISTData(self.trainImagesPath, self.trainLabelsPath)
		shuffle(self.trainData)
		self.testData = getMNISTData(self.testImagesPath, self.testLabelsPath)

		self.epoch = 0
		self.epochPos = 0

		self.iterationsPerPrint = 5000

		

	def train(self, iterations):

		startTime = time.time()
		currentTime = startTime		
		for i in range(iterations):
			
			#Get a new minibatch.  If at end of epoch, reshuffle data and start over.
			if self.epochPos + self.batchSize <= len(self.trainData):
				miniBatch = self.trainData[self.epochPos:self.epochPos+self.batchSize]
				self.epochPos += self.batchSize
			else:
				remainder = (self.epochPos + self.batchSize) - len(self.trainData)
				miniBatch = self.trainData[self.epochPos:]
				shuffle(self.trainData)
				miniBatch.extend(self.trainData[0:remainder])
				self.epochPos = remainder
				self.epoch += 1
		
			#Unzip the zipped images/label data.
			unzipped = list(zip(*miniBatch))
			images = Variable(torch.from_numpy((np.array(unzipped[0])/255).reshape(self.batchSize, self.channels, self.imageX, self.imageY)).type(self.floatType),requires_grad=False)
			labels = Variable(torch.from_numpy(np.array(unzipped[1])).type(self.longType), requires_grad=False)

			#Run net forward
			labelsPred = self.NN.feedForward(images,True)

			#Loss
			loss = torch.nn.functional.cross_entropy(labelsPred, labels)

			#Backpropagate and update weights
			loss.backward()

			#Update console
			if self.iterationsPerPrint > 0:
				if (i+1) % self.iterationsPerPrint == 0:
					timeSpent = time.time() - currentTime
					currentTime = time.time()	
					print("Completed training iteration " + str(i+1) + ".  Iteration loss: " + "{:3.2f}".format(loss.data[0]) + ". Time spent: " + "{:2.2f}".format(timeSpent))


			self.NN.updateWeights(self.step)
		return time.time() - startTime


	def test(self):

		correct = 0
		for datum in self.testData:

			#Unzip image and label data
			image = Variable(torch.from_numpy((np.array(datum[0])/255).reshape(1, self.channels, self.imageX, self.imageY)).type(self.floatType),requires_grad=False)
			label = Variable(torch.from_numpy(np.array(datum[1]).reshape(1)).type(self.longType),requires_grad=False)

			#Get net prediction
			_, labelPred = torch.max(self.NN.feedForward(image, False),1)


			#Check if correct
			if labelPred.data[0] == label.data[0]:
				correct += 1

		accuracy = correct / len(self.testData)
		
		return accuracy


if __name__ == "__main__":

	iterations = 100000
	batchSize = 100
	stepSize = 1e-3
	dataDirectory = "./MNIST_Data/"
	
	use_gpu = torch.cuda.is_available()

	print("Initializing network.")
	NN = CNNFixed.CNNFixed((MNISTNet.channels, MNISTNet.imageX, MNISTNet.imageY),(MNISTNet.classes,),use_gpu)

	print("Loading data.")
	net = MNISTNet(NN, batchSize, stepSize, dataDirectory, use_gpu)

	print("Training.")
	trainTime = net.train(iterations)
	print("Total time spent training: " + "{:6.0f}".format(trainTime))
	print("Testing.")
	accuracy = net.test()

	print("Accuracy after " + str(iterations) + " iterations: " + "{:2.2f}".format(accuracy))


	
	

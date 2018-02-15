'''
Functions for preprocessing Atari screen states.

Assumes input is a numpy array of shape (xLength, yLength, colorChannels).
'''

import numpy as np

class PreProcessor:

	def __init__(self, inputScreenShape):
		self.screenShape = inputScreenShape
		self.cropShape = None
		self.qGreyScale = None
		self.coarseGrainFactor = None
		self.maxValue = None
		self.inputMaxValue = None

	def outputShape(self):
		
		if self.cropShape is not None:
			shape = self.cropShape
		else:
			shape = self.inputScreenShape
	
		if self.coarseGrainFactor is not None:
			coarseShape = (int(shape[0] / self.coarseGrainFactor), int(shape[1] / self.coarseGrainFactor))
		else:
			coarseShape = (shape[0], shape[1])

		if qGreyScale is True:
			colorChannels = 1
		else:
			colorChannels = self.screenShape[2]

		return (coarseShape[0], coarseShape[1], colorChannels)

	def crop(self, observation, offset):
		outside = np.subtract(self.screenShape[0:2], self.cropShape)
		start = np.floor_divide(outside, 2)
		end = np.subtract(self.screenShape[0:2], start)
		#The floor_divide funtion will over-estimate the size if the number of the outside points is odd.  Subtract one if so.
		if outside[0] % 2 == 1:
			end[0] -= 1
		if outside[1] % 2 == 1:
			end[1] -= 1
		start += offset
		end += offset
		if start[0] < 0 or end[0] > observation.shape[0] or start[1] < 0 or end[1] > observation.shape[1]:
			raise ValueError("Offset has shifted cropped region off the side of the image.")
		return observation[start[0]:end[0],start[1]:end[1]]

	#Reduces to one color channel by averaging
	def greyScale(self, observation):
		return np.average(observation, 2).reshape((observation.shape[0], observation.shape[1], 1))

	
	def coarseGrain(self, observation):
		f = self.coarseGrainFactor
		size = np.floor_divide(observation.shape[0:2], f)
		output = np.array([np.average(observation[f*i:f*(i+1),f*j:f*(j+1),k]) for i in range(size[0]) for j in range(size[1]) for k in range(observation.shape[2])]).reshape((size[0],size[1],observation.shape[2]))
		return output
		
	def rescale(self, observation):
		if self.maxValue is None:
			return observation

		#Use the largest value in the input if no known max is provided
		if self.inputMaxValue is None:
			maxIn = np.max(observation)
		else:
			maxIn = self.inputMaxValue

		f = self.maxValue / maxIn
		return f * observation


	def process(self, observation, offset=(0,0)):
		if self.cropShape is not None:
			data = self.crop(observation, offset)
		else:
			data = observation

		if self.qGreyScale is True:
			data = self.greyScale(data)
		
		if self.coarseGrainFactor is not None:
			data = self.coarseGrain(data)

		if self.maxValue is not None:
			data = self.rescale(data)

		return data


if __name__ == "__main__":
	
	testSize = (4,4,3)
	p = PreProcessor(testSize)
	p.cropShape = (2,2)
	testData = np.random.random(testSize)
	print("Raw data sized " + str(testSize) + ":")
	print(testData)
	output = p.process(testData)
	print("Cropped to " + str(p.cropShape) + ":")
	print(output)

	p.coarseGrainFactor = 2
	print("Cropped to " + str(p.cropShape) + " then coarse grained by " + str(p.coarseGrainFactor) + ":")
	output = p.process(testData)
	print(output)
	p.coarseGrainFactor = None

	p.cropShape = (3,3)
	p.qGreyScale = True
	output = p.process(testData)
	print("Cropped to " + str(p.cropShape) + ":")
	print(output)

	testSize = (7,7,1)
	p.screenShape = testSize
	testData = np.random.random(testSize)
	print("Raw data sized " + str(testSize) + ":")
	print(testData)
	p.cropShape = (5,4)
	output = p.process(testData)	
	print("Cropped to " + str(p.cropShape) + ":")
	print(output)
	output = p.process(testData)
	offset = (1,1)
	print("Cropped to " + str(p.cropShape) + " with offset " + str(offset) + ":")
	output = p.process(testData, offset)
	print(output)

	p.coarseGrainFactor = 2
	print("Cropped to " + str(p.cropShape) + " then coarse grained by " + str(p.coarseGrainFactor) + ":")
	output = p.process(testData)
	print(output)

	p.coarseGrainFactor = None
	p.cropShape = None
	p.inputMaxValue = 1
	p.maxValue = 0.25
	print("Scaled down to a maximum of " + str(p.maxValue) + ":")
	output = p.process(testData)
	print(output)

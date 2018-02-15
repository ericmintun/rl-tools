'''
Functions for preprocessing batches of 2D, three color channel screen data.  Generally assumes data is of the form (..., colorChannel, y, x).
'''

import numpy as np

def crop(image, cropSize, offset=(0,0)):
	'''
	Will crop the screen to the specified size, with the specified offset from the center of the image.  If an odd number of pixels are cropped out, (0,0) offset will prefer to put one fewer cropped pixel on the left or on the top.  Will raise an error if the cropped region lies outside the image.
	
	Parameters:
	image -- a numpy array of two or more dimensions.  The crop will be performed in the last two dimensions.
	cropSize -- a 2-tuple specifying the Y and X sizes of the cropped image.
	offset -- a 2-tuple specifying the Y and X offsets of the crop from the center of the image.
	
	Output:
	Numpy array with the same number of dimensions as image.  All but the last two dimensions have the same length as image, while the last two have size cropSize.
	'''

	imageSize = image.shape[-2:]
	remainderSize = np.subtract(imageSize, cropSize)

	if remainderSize[0] < 0 or remainderSize[1] < 0:
		raise ValueError("Requested cropped image size is bigger than the image.")

	startPos = np.add(np.floor_divide(remainderSize, 2), offset)
	endPos = np.add(startPos, cropSize)

	if startPos[0] < 0 or endPos[0] > imageSize[0] or startPos[1] < 0 or endPos[1] > imageSize[1]:
		raise ValueError("The requested offset has pushed the cropping region outside the image.")

	return image[...,startPos[0]:endPos[0],startPos[1]:endPos[1]]



def greyScale(image, colorAxis=-3):
	'''
	Averages over a single axis of the input image, designed to reduce colored image to greyscale images.  By default assumes the color channel is the third from last axis.  The output numpy array has the same number of dimensions as the input array, but the length of the reduced dimension is 1.

	Parameters:
	image -- a numpy array of arbitrary dimension.
	colorAxis -- the axis to reduce over.  By default is the third from last.

	Output:
	A numpy array with the same number of dimensions as image.  The axis reduced over will have length 1.
	'''

	outputShape = list(image.shape)
	outputShape[colorAxis] = 1

	return np.average(image, colorAxis).reshape(outputShape)


def coarseGrain(image, factor):
	'''
	Coarse grains the image.  Looks at factor sized blocks of pixels, and outputs on pixel that is the average of the looked at pixels.If an image axis is not divisible by factor, this process will crop out the left over pixels on the right and bottom.  Coarse graining occurs in the last two axis.

	Parameters:
	image -- a numpy array of arbitrary dimension
	factor -- either a 2-tuple, which is (factorY, factorX), or a single number, for which the coarse graining is square.

	Output:
	A numpy array with the same number of dimensions as the image.  The last two axis has lengths floor(originalLength / factor).
	'''
	
	if type(factor) is tuple:
		f = factor
	else:
		f = (factor, factor)

	outputShape = list(image.shape)
	outputShape[-2] = image.shape[-2] // f[0]
	outputShape[-1] = image.shape[-1] // f[1]

	output = np.zeros(outputShape)

	for i in range(outputShape[-2]):
		for j in range(outputShape[-1]):
			output[...,i,j] = np.average(np.average(image[...,i*f[0]:(i+1)*f[0],j*f[1]:(j+1)*f[1]],-2),-1)
	
	return output


def rescale(image, newMaxValue, oldMaxValue=None):
	'''
	Rescales all values in an image.  If oldMaxValue is None, the largest value in the entire provided array will be rescaled to newMaxValue, and all other values will be rescaled accordingly.  If oldMaxValue is provided, this function literally just multiplies every value in the array by newMaxValue/oldMaxValue.

	Parameters:
	image -- a numpy array of arbitrary dimension
	newMaxValue -- the largest value a number in the new array is allowed to take.
	oldMaxValue -- the largest value a number could previously take.  If this is None, it will search for the maximum value in the numpy array.

	Output:
	A numpy array with the same number of dimensions as image.
	'''

	if oldMaxValue == None:
		maxIn = np.max(image)
	else:
		maxIn = oldMaxValue

	f = newMaxValue / maxIn
	return f * image


def moveColorChannel(image, newChannelPos='before'):
	'''
	Changes the order of the image array so that the color channel index comes before the pixel positions or after them.

	Parameters:
	image -- a numpy array of three or more dimensions.  Should be of the form (..., colorChannel, y, x) or (..., y, x, colorChannel).
	newChannelPos -- can be 'before' or 'after'.  If 'before', assumes input is of the form (..., y, x, colorChannel) and changes it to (..., colorChannel, y, x).  If 'after', does the reverse.

	Output:
	A numpy array of dimension (..., y, x, colorChannel) if newChannelPos is 'after', or (..., colorChannel, y, x) if newChannelPos is 'before'.
	'''

	if len(image.shape) < 3:
		raise ValueError("Input array must have at least three dimensions.")

	if newChannelPos == 'before':
		return np.swapaxes(np.swapaxes(image, -1, -3), -2, -1)
	elif newChannelPos == 'after':
		return np.swapaxes(np.swapaxes(image, -3, -1), -2, -3)
	else:
		raise ValueError("newChannelPos must be 'before' or 'after', but was neither.")


class Preprocessor:
	'''
	A class for wrapping preprocessing functions together.  Can be initialized with various processing parameters, and can perform that same preprocessing on every input.
	'''

	def __init__(self, 
		initialColorPos = 'after',
		newColorPos = None,
		cropSize=None,
		offset=None,
		qGreyScale=None,
		coarseGrainFactor=None,
		oldMaxValue=None,
		newMaxValue=None):
		'''
		Initialize the preProcessor with the processing parameters.  Setting any parameter to None will cause that operation to not be preformed.  Needs to know whether the color channel is the last or third to last position.  Cropping is performed before coarse graining.  Rescaling is performed after greyscaling.
		'''
		self.initialColorPos = initialColorPos
		self.newColorPos = newColorPos
		self.cropSize = cropSize
		self.qGreyScale = qGreyScale
		self.coarseGrainFactor = coarseGrainFactor
		self.oldMaxValue = oldMaxValue
		self.newMaxValue = newMaxValue
		self.offset = offset


	def outputShape(self, inputShape):
		'''
		Determines the shape of the output image assuming the parameters the preProcessor was initialized with.

		Parameters:
		inputShape -- a tuple of length at least three

		Output:
		A tuple the same length as the input.  Only the last three terms will be different.
		'''
		if self.initialColorPos == 'after':
			yPos = -3
			xPos = -2
			cPos = -1
		elif self.initialColorPos == 'before':
			yPos = -2
			xPos = -1
			cPos = -3
		else:
			raise ValueError("Invalid initial color channel position.")

		if self.cropSize != None:
			imageSize = self.cropSize
		else:
			imageSize = (inputShape[yPos], inputShape[xPos])

		if self.coarseGrainFactor != None:
			if type(self.coarseGrainFactor) == tuple:
				f = self.coarseGrainFactor
			else:
				f = (self.coarseGrainFactor, self.coarseGrainFactor)
			imageSize = tuple(np.floor_divide(imageSize, f))

		if self.qGreyScale == True:
			channels = 1
		else:
			channels = inputShape[cPos]

		if self.newColorPos == 'after':
			return (imageSize[0], imageSize[1], channels)
		elif self.newColorPos == 'before':
			return (channels, imageSize[0], imageSize[1])
		elif self.newColorPOs == None:
			if self.initialColorPos == 'after':
				return (imageSize[0], imageSize[1], channels)
			else:
				return (channels, imageSize[0], imageSize[1])
		else:
			raise ValueError("Invalid final color channel position.")


	def process(self,
		image,
		initialColorPos = 'default',
		newColorPos = 'default',
		cropSize='default',
		offset='default',
		qGreyScale='default',
		coarseGrainFactor='default',
		oldMaxValue='default',
		newMaxValue='default'):

		'''
		Performs the requested processing.  Any of the parameters set in the class can be overrun, if left as default the class values are used.
	
		Parameters:
		image -- numpy array of at least three dimensions.
	all other parameters -- see respective processing functions

		Output:
		Numpy array with the same number of dimensions as image.

		'''

		if initialColorPos != 'default':
			_initialColorPos = initialColorPos
		else:
			_initialColorPos = self.initialColorPos
		if newColorPos != 'default':
			_newColorPos = newColorPos
		else:
			_newColorPos = self.newColorPos
		if cropSize != 'default':
			_cropSize = cropSize
		else:
			_cropSize = self.cropSize
		if offset != 'default':
			_offset = offset
		else:
			_offset = self.offset
		if qGreyScale != 'default':
			_qGreyScale = qGreyScale
		else:
			_qGreyScale = self.qGreyScale
		if coarseGrainFactor != 'default':
			_coarseGrainFactor = coarseGrainFactor
		else:
			_coarseGrainFactor = self.coarseGrainFactor
		if oldMaxValue != 'default':
			_oldMaxValue = oldMaxValue
		else:
			_oldMaxValue = self.oldMaxValue
		if newMaxValue != 'default':
			_newMaxValue = newMaxValue
		else:
			_newMaxValue = self.newMaxValue

		
		#Move the color channel to 'before' to apply further preprocessing steps.
		if _initialColorPos == 'after':
			data = moveColorChannel(image, 'before')
		elif _initialColorPos == 'before':
			data = image
		else:
			raise ValueError("Invalid position provided for initial color channel.")

		if _qGreyScale == True:
			data = greyScale(data)

		if _newMaxValue != None:
			data = rescale(data, _newMaxValue, _oldMaxValue)

		if _cropSize != None:
			if _offset == None:
				_offset = (0,0)
			data = crop(data, _cropSize, _offset)

		if _coarseGrainFactor != None:
			data = coarseGrain(data, _coarseGrainFactor)

		#Move the color channel to the right place.  If newColorPos isn't provided, set the position back to the old position.
		if _newColorPos	== 'after':
			data = moveColorChannel(data, 'after')
		elif _newColorPos == None and _initialColorPos == 'after':
			data = moveColorChannel(data, 'after')
		elif _newColorPos != 'before' and _initialColorPos != 'before':
			raise ValueError("Invalid position provided for new color channel.")

		return data

		

		
		
	

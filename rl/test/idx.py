'''
A function for reading IDX data.  Outputs data as a numpy array of the shape specified in the IDX file.
'''


import struct
import numpy as np
import gzip

def splicer(data, spliceSize):
	for i in range(0,len(data),spliceSize):
		yield data[i:i+spliceSize]

def readHeader(header):
	magicNumbers = list(header)
	dataType = magicNumbers[2]
	numDim = magicNumbers[3]
	if dataType == 8:
		byteSize = 1
		unpackParam = '>B'
	elif dataType == 9:
		byteSize = 1
		unpackParam = '>b'
	elif dataType == 11:
		byteSize = 2
		unpackParam = '>h'
	elif dataType == 12:
		byteSize = 4
		unpackParam = '>i'
	elif dataType == 13:
		byteSize = 4
		unpackParam = '>f'
	elif dataType == 14:
		byteSize = 8
		unpackParam = '>d'
	else:
		raise IOError('Invalid data type specified in IDX file.')
		
	return numDim, byteSize, unpackParam		
		
def read(filename):
	with gzip.open(filename, 'rb') as f:
		magicNumber = f.read(4)
		numDim, byteSize, unpackParam = readHeader(magicNumber)
			
		arrayDims = []
		for i in range(numDim):
			arrayDims.append(struct.unpack('>i', f.read(4))[0])
			
		data = f.read()
		dataArray = np.array(list(map(lambda x : struct.unpack(unpackParam, x)[0], splicer(data,byteSize))))
		return np.reshape(dataArray, tuple(arrayDims))	

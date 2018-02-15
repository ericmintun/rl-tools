'''
Simple test cases for preprocessing image input.  Runs on MNIST data.  Uses matplotlib for output.
'''

import preprocess as pre
import numpy as np
import unittest

class TestPreProcessor(unittest.TestCase):

	def test_crop(self):
		uncropped = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]])
		#Test without offset
		croppedTest1 = np.array([[[6,7],[10,11]]]) 
		croppedTest1Crop = (2,2)
		croppedTest1Offset = (0,0)
		self.assertTrue(np.allclose(pre.crop(uncropped, croppedTest1Crop, croppedTest1Offset), croppedTest1))
		#Test offset
		croppedTest2 = np.array([[[11,12],[15,16]]])	
		croppedTest2Crop = (2,2)
		croppedTest2Offset = (1,1)
		self.assertTrue(np.allclose(pre.crop(uncropped, croppedTest2Crop, croppedTest2Offset), croppedTest2))
		#Test non-square factor regions and preference for top and left with odd remainders.
		croppedTest3 = np.array([[[2],[6],[10]]])
		croppedTest3Crop = (3,1)
		croppedTest3Offset = (0,0)
		self.assertTrue(np.allclose(pre.crop(uncropped, croppedTest3Crop, croppedTest3Offset), croppedTest3))
		with self.assertRaises(ValueError):
			#Fails when crop region is too big.
			croppedTest4Crop = (6,3)
			croppedTest4Offset = (0,0)
			pre.crop(uncropped, croppedTest4Crop, croppedTest4Offset)
			#Fails when offset is too large.
			croppedTest5Crop = (2,2)
			croppedTest5Offset = (2,1)
			pre.crop(uncropped, croppedTest5Crop, croppedTest5Offset)

	def test_coarseGrain(self):
		initial = np.array([[[1,0,1,0,1,0],[0,0,0,0,0,0],[0,1,0,1,0,1,],[1,1,1,1,1,1]]])
		#Test square factor sizes
		cgTest1 = np.array([[[1/4,1/4,1/4],[3/4,3/4,3/4]]])
		cgTestF1 = 2
		self.assertTrue(np.allclose(pre.coarseGrain(initial, cgTestF1), cgTest1))
		#Test non-square factor sizes
		cgTest2 = np.array([[[2/6,1/6],[4/6,5/6]]])
		cgTestF2 = (2,3)
		self.assertTrue(np.allclose(pre.coarseGrain(initial, cgTestF2), cgTest2))
		#Test cropped of left over pixels
		cgTest3 = np.array([[[3/9,3/9]]])
		cgTestF3 = 3
		self.assertTrue(np.allclose(pre.coarseGrain(initial, cgTestF3), cgTest3))
		

	def test_greyScale(self):
		initial = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
		#Test regular color axis
		greyScaleTest1 = np.array([[[6/2,8/2],[10/2,12/2]]])
		self.assertTrue(np.allclose(pre.greyScale(initial),greyScaleTest1))
		#Test different color axis
		greyScaleTest2 = np.array([[[3/2],[7/2]],[[11/2],[15/2]]])
		greyScaleTest2Axis = -1
		self.assertTrue(np.allclose(pre.greyScale(initial, greyScaleTest2Axis),greyScaleTest2))

	def test_rescale(self):
		initial = np.array([[[2,4],[6,8]]])
		#Test with implicit max
		rescaleTest1 = np.array([[[1,2],[3,4]]])
		rescaleTest1NewMax = 4
		self.assertTrue(np.allclose(pre.rescale(initial, rescaleTest1NewMax),rescaleTest1))
		#Test with explicit max
		rescaleTest2 = np.array([[[1/2,1],[3/2,2]]])
		rescaleTest2NewMax = 4
		rescaleTest2OldMax = 16
		self.assertTrue(np.allclose(pre.rescale(initial, rescaleTest2NewMax, rescaleTest2OldMax),rescaleTest2))

	def test_moveColorChannel(self):
		initial = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
		#Before to after
		moveColorTest1Pos = 'after'
		moveColorTest1 = np.array([[[1,5],[2,6]],[[3,7],[4,8]]])
		self.assertTrue(np.allclose(pre.moveColorChannel(initial, moveColorTest1Pos),moveColorTest1))
		#After to before
		moveColorTest2Pos = 'before'
		moveColorTest2 = np.array([[[1,3],[5,7]],[[2,4],[6,8]]])
		self.assertTrue(np.allclose(pre.moveColorChannel(initial, moveColorTest2Pos),moveColorTest2))

	def test_preProcess(self):

		initial = np.array(
			[[  [50,50,50],   [75, 0, 0],   [ 0, 0, 0],   [25,25,75]  ],
			 [  [ 0, 0, 0],   [25, 0,25],   [75,50,25],   [25,25,75]  ],
			 [  [50,50,50],   [75,25,50],   [50,50,50],   [ 0, 0, 0]  ],
			 [  [50,50,50],   [75, 0, 0],   [ 0, 0, 0],   [25,25,75]  ]])

		p = pre.preProcessor(
			initialColorPos = 'after',
			newColorPos = 'before',
			cropSize = (2,2),
			offset = (0,0),
			qGreyScale = True,
			coarseGrainFactor = 2,
			oldMaxValue = 100,
			newMaxValue = 1)

		#After crop and grey scale.
		testIntermediate1 = np.array(
			[[  [ 50/3, 150/3],
			    [150/3, 150/3]]])   

		#After rescale.
		testIntermediate2 = np.array(
			[[  [ 50/300, 150/300],
			    [150/300, 150/300]]])

		#Final.
		test = np.array(
			[[[5/12]]])

		#Test preprocessor
		self.assertTrue(np.allclose(p.process(initial),test))

		#Test overriding using intermediate steps
		processed2 = p.process(initial,
			coarseGrainFactor = None,
			newMaxValue = None)
		self.assertTrue(np.allclose(processed2, testIntermediate1))
		processed3 = p.process(initial,
			coarseGrainFactor = None)
		self.assertTrue(np.allclose(processed3, testIntermediate2))



if __name__ == "__main__":
	unittest.main()

'''
Tests the preprocessors.py, which convert raw observation data in a
format prepared for input into a network.
'''

import unittest
import utils.image as img
import preprocessors as pre
import numpy as np

class TestPreprocessors(unittest.TestCase):

    def test_PixelPreprocessor(self):
        frame1 = np.array([[[1, 2], 
                            [3, 4]],

                           [[5, 6], 
                            [7, 8]]])

        frame2 = np.array([[[4, 4], 
                            [4, 4]],

                           [[4, 4], 
                            [4, 4]]])

        frame3 = np.array([[[1, 7], 
                            [6, 9]],

                           [[2, 1], 
                            [2, 6]]])

        #Test default behavior
        p1 = pre.PixelPreprocessor(
                image_processor=None,
                frame_number=1,
                frame_step=1,
                frame_copy=True,
                flatten_channels=True)

        self.assertTrue(np.array_equal(p1.process(frame1), frame1))
        self.assertTrue(np.array_equal(p1.process(frame2), frame2))
        self.assertTrue(np.array_equal(p1.process(frame3), frame3))

        #Test saving past frames
        p2 = pre.PixelPreprocessor(
                image_processor=None,
                frame_number=2,
                frame_step=1,
                frame_copy=True,
                flatten_channels=True)

        output1 = np.array([[[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]],

                            [[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]]])

        output2 = np.array([[[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]],

                            [[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]]])
        
        output3 = np.array([[[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]],

                            [[1, 7], 
                             [6, 9]],

                            [[2, 1], 
                             [2, 6]]])


        output4 = np.array([[[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]]])

        self.assertTrue(np.array_equal(p2.process(frame1), output1))
        self.assertTrue(np.array_equal(p2.process(frame2), output2))
        self.assertTrue(np.array_equal(p2.process(frame3), output3))
        p2.reset_episode()
        self.assertTrue(np.array_equal(p2.process(frame2), output4))
        
        #Test zeroing at beginning of episode
        p3 = pre.PixelPreprocessor(
                image_processor=None,
                frame_number=2,
                frame_step=1,
                frame_copy=False,
                flatten_channels=True)

        output5 = np.array([[[0, 0], 
                             [0, 0]],

                            [[0, 0], 
                             [0, 0]],

                            [[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]]])

        self.assertTrue(np.array_equal(p3.process(frame1), output5))

        #Test frame skip
        p4 = pre.PixelPreprocessor(
                image_processor=None,
                frame_number=2,
                frame_step=2,
                frame_copy=False,
                flatten_channels=True)

        output6 = np.array([[[0, 0], 
                             [0, 0]],

                            [[0, 0], 
                             [0, 0]],

                            [[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]]])

        output7 = np.array([[[0, 0], 
                             [0, 0]],

                            [[0, 0], 
                             [0, 0]],

                            [[1, 7], 
                             [6, 9]],

                            [[2, 1], 
                             [2, 6]]])

        output8 = np.array([[[1, 2], 
                             [3, 4]],

                            [[5, 6], 
                             [7, 8]],

                            [[4, 4], 
                             [4, 4]],

                            [[4, 4], 
                             [4, 4]]])

        self.assertTrue(np.array_equal(p4.process(frame1), output6))
        self.assertTrue(np.array_equal(p4.process(frame3), output7))
        self.assertTrue(np.array_equal(p4.process(frame2), output8))

        #Test no channel flatten
        p5 = pre.PixelPreprocessor(
                image_processor=None,
                frame_number=2,
                frame_step=1,
                frame_copy=False,
                flatten_channels=False)

        output9 = np.array([[[[0, 0], 
                              [0, 0]],

                             [[0, 0], 
                              [0, 0]]],

                            [[[1, 2], 
                              [3, 4]],

                             [[5, 6], 
                              [7, 8]]]])

        self.assertTrue(np.array_equal(p5.process(frame1), output9))

        #Test image processing
        i = img.make(qGreyScale=True,coarseGrainFactor=2)
        p6 = pre.PixelPreprocessor(
                image_processor=i,
                frame_number=2,
                frame_step=1,
                frame_copy=False,
                flatten_channels=True)
        
        output10 = np.array([[[0]],[[4.5]]])

        self.assertTrue(np.array_equal(p6.process(frame1), output10))

if __name__ == "__main__":
    unittest.main()

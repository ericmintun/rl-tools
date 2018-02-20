'''
Tests for postprocessors.py
'''

import unittest
import numpy as np
import torch as t
from torch.autograd import Variable
import postprocessors as post

class TestPostprocessor(unittest.TestCase):

    def test_DiscreteQPostprocessor(self):
        
        ftype = t.FloatTensor
        ltype = t.LongTensor
        input_array = np.array([[1,2,3,4,5],
                                [0,4,1,2,2],
                                [5,0,0,1,3],
                                [3,1,5,2,1]])
        input = Variable(t.from_numpy(input_array).type(ftype),
                  requires_grad=True)
        p1 = post.DiscreteQPostprocessor()


        #Test receiving best Q values as output as well
        output1 = t.from_numpy(np.array([4,1,0,2])).type(ltype)
        output2 = t.from_numpy(np.array([5,4,5,5])).type(ftype)
        result1, result2 = p1.best_action(input, output_q=True)
        self.assertTrue(t.equal(result1.data, output1))
        self.assertTrue(t.equal(result2.data, output2))

        #Test without receiving Q
        result3 = p1.best_action(input)
        self.assertTrue(t.equal(result3.data, output1))




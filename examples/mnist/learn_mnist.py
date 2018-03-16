'''
A quick framework for running a NN against MNIST.  The only preprocessing done is to rescale the input to have a max value of 1.
'''

import numpy as np
import mnist.idx as idx
from random import shuffle
import rl.utils.image as pre
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def get_data(images_filename, labels_filename):
    images = idx.read(images_filename)
    labels = idx.read(labels_filename)
    if len(images) != len(labels):
        raise ValueError("The number of labels and number of images do not match.")
    return list(zip(images, labels))


class MNISTNet:

    input_y_size = 28
    input_x_size = 28
    color_channels = 1
    num_classes = 10
    input_max = 256
    input_rescaled_max = 1
    train_images_filename = 'train-images.idx3-ubyte.gz'
    train_labels_filename = 'train-labels.idx1-ubyte.gz'
    test_images_filename = 't10k-images.idx3-ubyte.gz'
    test_labels_filename = 't10k-labels.idx1-ubyte.gz'

    def __init__(self, NN, post, batch_size=50, step_size=1e-5,
                 MNIST_directory = './MNIST_data/', 
                 use_gpu = False, iter_per_print=None):
        self.NN = NN
        self.post = post
        self.batch_size = batch_size
        self.step_size = step_size
        self.p = pre.make(initialColorPos = 'before',
            oldMaxValue = self.input_max,
            newMaxValue = self.input_rescaled_max)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.NN.parameters(), self.step_size)

        if use_gpu == False:
            self.float_type = torch.FloatTensor
            self.long_type = torch.LongTensor
        else:
            self.float_type = torch.cuda.FloatTensor
            self.long_type = torch.cuda.LongTensor    

        self.train_images_path = MNIST_directory + self.train_images_filename
        self.train_labels_path = MNIST_directory + self.train_labels_filename
        self.test_images_path = MNIST_directory + self.test_images_filename
        self.test_labels_path = MNIST_directory + self.test_labels_filename

        self.train_data = get_data(self.train_images_path, self.train_labels_path)
        shuffle(self.train_data)
        self.test_data = get_data(self.test_images_path, self.test_labels_path)

        self.epoch = 0
        self.current_pos = 0

        self.iter_per_print = iter_per_print

    def prep_data(self, raw_data, batch_size):
        if batch_size > 1:
            unzipped = list(zip(*raw_data))
        else:
            unzipped = [[raw_data[0]], [raw_data[1]]]

        input_shape = (batch_size, self.color_channels, self.input_y_size, self.input_x_size)
        images_array = self.p.process(np.array(unzipped[0])).reshape(input_shape)
        images = Variable(torch.from_numpy(images_array).type(self.float_type), requires_grad=False)
        labels_array = np.array(unzipped[1])
        labels = Variable(torch.from_numpy(labels_array).type(self.long_type),requires_grad=False)
        return images, labels


    def minibatch(self):
        if self.current_pos + self.batch_size <= len(self.train_data):
            minibatch = self.train_data[self.current_pos:self.current_pos+self.batch_size]
            self.current_pos += self.batch_size
        else:
            remainder = (self.current_pos + self.batch_size) - len(self.train_data)
            minibatch = self.train_data[self.current_pos:]
            shuffle(self.train_data)
            minibatch.extend(self.train_data[0:remainder])
            self.current_pos = remainder
            self.epoch += 1

        return minibatch

    
    def train(self, iterations, time_cutoff = None):
        self.NN.train(True)
        start_time = time.time()
        current_time = start_time

        for i in range(iterations):

            images, labels = self.prep_data(self.minibatch(), self.batch_size)
            pred = self.post.predictions(self.NN.forward(images))
            loss = self.cross_entropy.forward(pred,labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            #Update console
            if self.iter_per_print != None:
                if (i+1) % self.iter_per_print == 0:
                    time_spent = time.time() - current_time
                    current_time = time.time()    
                    print("Completed training iteration " + str(i+1) + ".  Iteration loss: " + "{:3.2f}".format(loss.data[0]) + ". Time spent: " + "{:2.2f}".format(time_spent))

            if time_cutoff is not None and time.time() - current_time > time_cutoff:
                break

        self.NN.train(False)
        return time.time() - start_time, i+1

    
    def test(self):
        
        self.NN.train(False)
        correct = 0
        for datum in self.test_data:
            image, label = self.prep_data(datum, 1)
            _, pred = torch.max(self.post.predictions(self.NN.forward(image)),1)
            
            if pred.data[0] == label.data[0]:
                correct += 1

        accuracy = correct / len(self.test_data)
        return accuracy





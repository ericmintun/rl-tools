'''
Tests RL_CNN.py by running it again MNIST.  Demands a low but not random accuracy of >0.5, since the
goal is really just to test that the model is training at all.  Maximum running time is ~15 minutes.
'''

import torch
import test.learn_mnist as mnist
import rl_cnn as cnn
import unittest

class TestRLCNN(unittest.TestCase):

	def test_mnist(self):

		use_gpu = torch.cuda.is_available()

		#Set up the neural network
		image_y_size = mnist.MNISTNet.input_y_size
		image_x_size = mnist.MNISTNet.input_x_size
		color_channels = mnist.MNISTNet.color_channels
		classes = mnist.MNISTNet.num_classes

		input_size = (color_channels, image_y_size, image_x_size)
		output_size = (classes,)

		print("Initializing network.")
		net = cnn.RL_CNN(input_size, output_size)
		if use_gpu:
			net.cuda()

		#Set up the trainer
		batch_size = 100
		step_size = 1e-3
		data_directory = './test/MNIST_data/'
		iter_per_print = 1000

		print("Loading data.")
		trainer = mnist.MNISTNet(net, batch_size, step_size, data_directory, use_gpu, iter_per_print)

		#Training parameters
		iterations = 20000
		time_cutoff = 1000

		#Train!
		print("Training.")
		time_taken, iter_completed = trainer.train(iterations, time_cutoff)

		if iter_completed == iterations:
			print("All " + str(iterations) + " iterations completed in a total time of " + "{:2.2f}".format(time_taken) + " seconds.")
		else:
			print("Only " + str(iter_completed) + " iterations completed after time cutoff of " + "{:2.2f}".format(time_cutoff) + " seconds reached.")

		#Test!
		print("Testing.")
		accuracy = trainer.test()
		print("Test accuracy: " + "{:1.2f}".format(accuracy))
		self.assertTrue(accuracy > 0.5)


if __name__ == "__main__":
	unittest.main()

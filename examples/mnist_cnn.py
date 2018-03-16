'''
A quick test of rl_cnn against MNIST.
'''

import torch
import rl
import mnist.learn_mnist as mnist
import nets.rl_cnn as cnn
import nets.rl_capsule as cap
import sys


def main():

    if len(sys.argv) > 1:
        net_type = sys.argv[1]
    else:
        net_type = "CNN"

    use_gpu = torch.cuda.is_available()

    #Set up the neural network
    image_y_size = mnist.MNISTNet.input_y_size
    image_x_size = mnist.MNISTNet.input_x_size
    color_channels = mnist.MNISTNet.color_channels
    classes = mnist.MNISTNet.num_classes

    input_size = (color_channels, image_y_size, image_x_size)
    output_size = (classes,)

    print("Initializing network.")
    if net_type == "CNN":
        net = cnn.RL_CNN(input_size, output_size)
        post = rl.postprocessors.PredictionPostprocessor()
        batch_size = 100
        iter_per_print = 1000
    elif net_type == "Capsule":
        net = cap.RL_Capsule(input_size, output_size)
        post = rl.postprocessors.CapsuleBasicPostprocessor()
        batch_size = 32
        iter_per_print = 100
    else:
        raise ValueError("Unrecognized net type specified.")

    if use_gpu:
        net.cuda()

    #Set up the trainer
    step_size = 1e-3
    data_directory = './mnist/MNIST_data/'

    print("Loading data.")
    trainer = mnist.MNISTNet(net, post, batch_size, step_size, data_directory, use_gpu, iter_per_print)

    #Training parameters
    iterations = 20000
    time_cutoff = None

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


if __name__ == "__main__":
    main()

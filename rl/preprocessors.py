'''
Classes for turning raw output of an environment into input for a
network.

Currently preprocessors are designed to work with single state inputs, 
since the assumption is that replay memory and batching will occur after
preprocessing.  This may not be a good long term assumption.

Preprocessors do not currently convert into and out of torch variables.

All preprocessors should have the following methods:

process(input_state) : which takes the raw observation and outputs the
  processed observation.  The form of the input and output can vary from
  processor to processor.

batchify(states, processed=True) : takes in a list of states, and
  prepares them as a batch input to the network.  If 'processed' is
  True, assumes each element in the list is the output of the process
  method.  If 'processed' is False, assumes each element is a raw
  observation.  Not all preprocessors can batch raw observations, since
  it is possible necessary information for processing each element has
  been lost.  Should also be able to handle a single state not in a list.

reset_epsiode() : tells the processor that a new episode has begun.

'''

import numpy as np


class PixelPreprocessor:
    '''
    Preprocesses states where the observation is assumed to be raw pixel
    data in the form of a 3D numpy array.  Can remember old states
    incase the network input consists of several past frames.

    Initialization arguments:
    image_processor (ImgProcessor) : class that crops/rescales/etc an
      input image.  Can be any class with a process(image) method where
      'image' is a 3D numpy array, whose output is a 3D numpy array of 
      the form (color_channels, image_y, image_x).  If None, applies no
      image processing and the input to process must have the form
      (color_channels, image_y, image_x)
    frame_number (int) : the number of past frames to concatenate
    frame_step (int) : output past frames step back this many frames for 
      every frame remembered.
    frame_copy (bool) : if true, at the beginning of an episode past
      frames will be filled with copies of the first frame.  If false,
      past frames will be set to all zeros.
    flatten_channels (bool) : if true, color channels and frames are
      flattened into a single index, and output is a 3D numpy array of
      the form (channels, image_y, image_x).  If false, output is a 4D 
      array of the form (frames, color_channels, image_y, image_x).

    Methods:
    process(image) : Turns an input image in the form of a 3D numpy
      array into a processed image, which is either a 3D numpy array or
      4D numpy array depending on flatten_channels.

    reset_episode() : Tells the preprocessor a new episode has begun and
      it should forget any remembered past states.

    batchify(states, processed=True) : Turns a list of numpy arrays into
      a single numpy array of one higher dimension.  'processed' can
      only be False if frame_number = 1, since otherwise past frame data
      has been lost.

    ouput_shape(input_shape) : Given the shape of the input image as a
      3-tuple, returns the shape of the processed image as a tuple.
    '''


    def __init__(self,
            image_processor=None,
            frame_number=1,
            frame_step=1,
            frame_copy=True,
            flatten_channels=True):
        self.image_processor = image_processor
        self.frame_number = frame_number
        self.frame_step = frame_step
        self.frame_copy = frame_copy
        self.flatten_channels = flatten_channels

        self.frame_buffer = []
        self.buffer_size = self.frame_number * self.frame_step
        self.new_episode = True

    def process(self, image):
        if self.image_processor != None:
            reduced_image = self.image_processor.process(image)
        else:
            reduced_image = image

        if self.frame_number != None and self.frame_number > 1:
            # If a new episode, fill buffer.
            if self.new_episode:
                self.new_episode = False
                if self.frame_copy:
                    self.frame_buffer = [reduced_image 
                                            for i in range(self.buffer_size)]
                else:
                    self.frame_buffer = [np.zeros(reduced_image.shape) 
                                            for i in range(self.buffer_size)]
            #Add new image to buffer
            for i in range(self.buffer_size - 1):
                self.frame_buffer[i] = self.frame_buffer[i+1]
            self.frame_buffer[-1] = reduced_image

            #Build numpy array out of buffer.
            output = np.array(self.frame_buffer[self.frame_step-1::self.frame_step])

            #Flatten if desired.
            if self.flatten_channels:
                old_shape = output.shape
                new_shape = (output.shape[0] * output.shape[1], 
                             output.shape[2], output.shape[3])
                output = output.reshape(new_shape)
        else:
            output = reduced_image

        return output

    def batchify(self, states, processed=True):
        if processed == True:
            if type(states)==list:
                return np.array(states)
            else:
                return np.array([states]) #Assure single states have correct shape.
        if self.frame_number != 1 and self.frame_number != None:
            raise ValueError("Cannot process batched states since processing requires old frames which have been lost.")
        return np.array([self.process(state) for state in states])


    def reset_episode(self):
        self.new_episode = True

    def output_shape(self, input_shape):
        #Just runs process on zeros of the correct shape.
        dummy_input = np.zeros(input_shape)
        dummy_output = self.process(dummy_input)
        return dummy_output.shape

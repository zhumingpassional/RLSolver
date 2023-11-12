import numpy as np
import tensorflow as tf
from Tensordot2 import tensordot
import tf_slim as slim
#######################################################################################################
"""
Code by Mohamed Hibat-Allah
Title : An implementation of the two dimensional Tensorized RNN cell
"""
#######################################################################################################

class MDTensorizedRNNCell(tf.contrib.rnn.RNNCell):
    """The 2D Tensorized RNN cell.
    """
    def __init__(self, num_units = None, activation = None, name=None, dtype = None, reuse=None):
        super(MDTensorizedRNNCell, self).__init__(_reuse=reuse, name=name)
        # save class variables
        self._num_in = 2
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units
        self.activation = activation

        # set up input -> hidden connection
        self.W = tf.get_variable("W_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)

        self.b = tf.get_variable("b_"+name, shape=[num_units],
                                    initializer=slim.xavier_initializer(), dtype = dtype)

    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, states):

        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat(states, 1),tf.concat(inputs,1))
        # prepare input linear combination
        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, num_units]

        preact = state_mul + self.b

        output = self.activation(preact) # [batch_sz, num_units] C

        new_state = output

        return output, new_state

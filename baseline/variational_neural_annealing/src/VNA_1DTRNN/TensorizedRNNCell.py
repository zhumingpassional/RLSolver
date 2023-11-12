import numpy as np
import tensorflow as tf
from Tensordot2 import tensordot
import tf_slim as slim

"""
Code by Mohamed Hibat-Allah
Title : An implementation of the one dimensional Tensorized RNN cell
"""

class TensorizedRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """The 1D Tensorized RNN cell.
    """

    def __init__(self, num_units = None, num_in = 2, activation = None, name=None, dtype = None, reuse=None):
        super(TensorizedRNNCell, self).__init__(_reuse=reuse, name=name)
        # save class variables
        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units
        self.activation = activation

        # set up input -> hidden connection
        self.W = tf.compat.v1.get_variable("W_"+name, shape=[num_units, num_units, self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)
        
        self.bh = tf.compat.v1.get_variable("bh_"+name, shape=[num_units],
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

    def call(self, inputs, state):

        inputstate_mul = tf.einsum('ij,ik->ijk', state,inputs)

        # prepare input linear combination
        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, num_units]

        preact = state_mul + self.bh

        output = self.activation(preact) # [batch_sz, num_units] C

        new_state = output

        return output, new_state

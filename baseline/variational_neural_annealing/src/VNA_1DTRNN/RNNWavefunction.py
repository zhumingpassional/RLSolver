import tensorflow as tf
import numpy as np
import random

"""
This implementation is based on RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Description: Here, we define the 1D RNNwavefunction class, which contains the sample method
that allows to sample configurations autoregressively from the RNN and
the log_probability method which allows to estimate the log-probability of a set of configurations.
More details are in https://arxiv.org/abs/2101.10154.
"""

class RNNWavefunction(object):
    def __init__(self,systemsize,cell=None,activation=tf.nn.relu,units=[10],scope='RNNWavefunction', seed = 111):
        """
            systemsize:  int
                         number of sites
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            activation:  activation of the RNN cell
            seed:        pseudo-random number generator
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of sites of the 1D chain

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        dim_inputs = [2]+units[:-1] #dim of inputs for each layer in the RNN


        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

                #Defining RNN cells with site-dependent parameters
                self.rnn=[tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[i],num_in = dim_inputs[i], activation = activation,name='RNN_{0}{1}'.format(i,n), dtype = tf.float64) for i in range(len(units))]) for n in range(self.N)]
                self.dense = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='RNNWF_dense_{0}'.format(n), dtype = tf.float64) for n in range(self.N)]

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension

            ------------------------------------------------------------------------
            Returns:         a tuple (samples,log-probs)

            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (numsamples,)
                             the log-probability of each sample
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                samples = []
                probs=[]

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_state=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn[n](inputs, rnn_state)
                    output=self.dense[n](rnn_output)
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,])
                    probs.append(output)
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

            self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

            probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.samples, self.log_probs

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,system-size)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            inputs=tf.zeros((self.numsamples, self.inputdim), dtype=tf.float64)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs=[]

                rnn_state=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn[n](inputs, rnn_state)
                    output=self.dense[n](rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

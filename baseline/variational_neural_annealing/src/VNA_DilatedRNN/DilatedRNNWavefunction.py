import tensorflow as tf
import numpy as np
import random

"""
This implementation is an adaptation of RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Here, we define the Dilated RNNwavefunction class, which contains the sample method
that allows to sample configurations autoregressively from the RNN and
the log_probability method which allows to estimate the log-probability of a set of configurations.
We also note that the dilated connections between RNN cells allow to take care of the long-distance
dependencies between spins more efficiently as explained in https://arxiv.org/abs/2101.10154.
"""

class DilatedRNNWavefunction(object):
    def __init__(self,systemsize,cell=tf.keras.layers.SimpleRNNCell,activation=tf.nn.relu,units=[2],scope='DilatedRNNwavefunction', seed = 111):
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
        self.numlayers = len(units)
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                tf.set_random_seed(seed)  # tensorflow pseudo-random generator

                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(n)+str(i),dtype=tf.float64) for n in range(self.N)] for i in range(self.numlayers)]
                self.dense = [tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(n)) for n in range(self.N)] #Define the Fully-Connected layer followed by a Softmax

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
            Returns:
            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            samples = []
            probs = []
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output=self.dense[n](rnn_output)
                    probs.append(output)
                    sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)

        probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.samples,self.log_probs

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:
            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize)
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

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                probs=[]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output=self.dense[n](rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

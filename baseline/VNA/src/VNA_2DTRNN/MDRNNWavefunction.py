import tensorflow as tf
import numpy as np
import random

"""
This implementation is based on RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Here, we define the 2D RNNwavefunction class, which contains the sample method
that allows to sample configurations autoregressively from the RNN and
the log_probability method which allows to estimate the log-probability of a set of configurations.
More details are in https://arxiv.org/abs/2101.10154.
"""

class MDRNNWavefunction(object):
    def __init__(self,systemsize_x = None, systemsize_y = None,cell=None,activation=None,num_units = None,scope='RNNWavefunction',seed = 111):
        """
            systemsize_x, systemsize_y:  int
                         number of sites in x, y directions
            cell:        a tensorflow RNN cell
            num_units:   int
                         number of memory units
            scope:       str
                         the name of the name-space scope
            activation:  activation of the RNN cell
            seed:        pseudo-random number generator
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x
        self.Ny=systemsize_y

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):

              tf.set_random_seed(seed)  # tensorflow pseudo-random generator

              #Defining the 2D Tensorized RNN cell with non-weight sharing
              self.rnn=[cell(num_units = num_units, activation = activation, name="rnn_"+str(0)+str(i),dtype=tf.float64) for i in range(self.Nx*self.Ny)]
              self.dense = [tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(i), dtype = tf.float64) for i in range(self.Nx*self.Ny)]

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

            samples:         tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (numsamples,)
                             the log-probability of each sample
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):

                #Initial input to feed to the lstm

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.numsamples=numsamples


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}

                for ny in range(self.Ny): #Loop over the boundaries for initialization
                    if ny%2==0:
                        nx = -1
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                    if ny%2==1:
                        nx = self.Nx
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


                for nx in range(self.Nx): #Loop over the boundaries for initialization
                    ny = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                #Making a loop over the sites with the 2DRNN
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)


        self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,0,1])

        probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,0,1,3])
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)


        return self.samples,self.log_probs


    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x, systemsize_y)
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

            #Initial input to feed to the lstm
            self.outputdim=self.inputdim


            samples_=tf.transpose(samples, perm = [1,2,0])
            rnn_states = {}
            inputs = {}

            for ny in range(self.Ny): #Loop over the boundaries for initialization
                if ny%2==0:
                    nx = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                if ny%2==1:
                    nx = self.Nx
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


            for nx in range(self.Nx): #Loop over the boundaries for initialization
                ny = -1
                rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                #Making a loop over the sites with the 2DRNN
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                            output=self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

            probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,0,1,3])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)

            return self.log_probs

import tensorflow as tf
import numpy as np
import random

import os
import time
from math import ceil
import argparse

from DilatedRNN import DilatedRNN
from DilatedRNN_WeightSharing import DilatedRNN_WeightSharing
from Helper_functions import *

"""
This implementation is an adaptation of RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Here, we define the Dilated RNN class with weight sharing, which contains the sample method
that allows to sample configurations autoregressively from the RNN and
the log_probability method which allows to estimate the log-probability of a set of configurations.
We also note that the dilated connections between RNN cells allow to take care of the long-distance
dependencies between spins more efficiently as explained in https://arxiv.org/abs/2101.10154.
"""

class DilatedRNN_WeightSharing(object):
    def __init__(self,systemsize,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell,activation=tf.nn.relu,units=[2],scope='DilatedRNN', seed = 111):
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
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=[cell(num_units = units[i], activation = activation,name="rnn_"+str(i),dtype=tf.float64) for i in range(self.numlayers)]
                self.dense = tf.compat.v1.layers.Dense(self.N,activation=tf.nn.softmax,name='DRNN_dense') #Define the Fully-Connected layer followed by a Softmax

    def projection(self,output,cities_queue):
        projected_output = cities_queue*output
        projected_output = projected_output/(tf.reshape(tf.norm(tensor=projected_output, axis = 1, ord=1), [self.numsamples,1])) + 1e-30 #l1 normalizing
        return projected_output

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

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                cities_queue = tf.ones((numsamples,self.inputdim), dtype = tf.float64)
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        rnn_states.append(self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i](rnn_output, rnn_states[i])

                    output=self.dense(rnn_output)
                    output = self.projection(output, cities_queue) #Projection of probability to construct a valid tour at the end
                    probs.append(output)
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)
                    cities_queue = cities_queue - inputs

        probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

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
            cities_queue = tf.ones((self.numsamples,self.inputdim), dtype = tf.float64)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs=[]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        rnn_states.append(self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for n in range(self.N):

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i](rnn_output, rnn_states[i])

                    output=self.dense(rnn_output)
                    output = self.projection(output, cities_queue)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])
                    cities_queue = cities_queue - inputs

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

class DilatedRNN(object):
    def __init__(self,systemsize,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell,activation=tf.nn.relu,units=[2],scope='DilatedRNN', seed = 111):
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
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(n)+str(i),dtype=tf.float64) for n in range(self.N)] for i in range(self.numlayers)]
                self.dense = [tf.compat.v1.layers.Dense(self.N,activation=tf.nn.softmax,name='wf_dense'+str(n)) for n in range(self.N)] #Define the Fully-Connected layer followed by a Softmax

    def projection(self,output,cities_queue):
        projected_output = cities_queue*output
        projected_output = projected_output/(tf.reshape(tf.norm(tensor=projected_output, axis = 1, ord=1), [self.numsamples,1])) + 1e-30 #l1 normalizing
        return projected_output

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
            samples_onehot = []
            probs = []

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                cities_queue = tf.ones((numsamples,self.inputdim), dtype = tf.float64)
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
                    output = self.projection(output, cities_queue) #Projection of probability to construct a valid tour at the end
                    probs.append(output)
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)
                    cities_queue = cities_queue - inputs

        probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

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
            cities_queue = tf.ones((self.numsamples,self.inputdim), dtype = tf.float64)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
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
                    output = self.projection(output, cities_queue)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])
                    cities_queue = cities_queue - inputs

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs


"""
__________________________________________________________________________________________________________________________________________________________
__________________________________________________________________________________________________________________________________________________________
__________________________________________________________________________________________________________________________________________________________
"""

#!/usr/bin/python
# -*- coding: utf-8 -*-

#----------------------------------------------------------------------------------------------------------
"""
This implementation is an adaptation of RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Title : Implementation of Variational Neural Annealing for the TSP
Description : TSP
"""
#-----------------------------------------------------------------------------------------------------------

class vca:

    def __init__(self, N, coordinates, n_warmup, n_anneal, n_train, T0):
        """
        N - system size of optimization problem
        coordinates - coordinates of the N cities in the format: np.array([[x1,y1], .., [xN, yN]])
        n_warmup - number of warmup iterations
        n_anneal - temperature annealing iterations
        n_train - RNN model training iterations
        T0 - initial temperature
        """
        #Seeding for reproducibility purposes
        seed = 111
        tf.compat.v1.reset_default_graph()
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

        # VCA hyperparameters
        self.num_warmup = n_warmup
        self.num_anneal = n_anneal
        self.num_train = n_train
        self.T0 = T0
        # RNN architecture hyperparameters
        self.N = N
        self.numlayers = np.int32(np.ceil(np.log2(N)))    # number of layers in the Dilated RNN architecture set to log2(N)
        self.numunits = 40    # size of RNN hidden units
        self.units = [self.numunits]*self.numlayers # defines the number of hidden units at each layer (for Dilated)
        self.numsamples = 50    # number of training samples
        self.lr = np.float64(1e-3)      # learning rate during backpropagation
        self.activation_function = tf.nn.elu
        self.weight_sharing = "True"    # every RNN bnlock has the same set of RNN parameters. Set to "False" for dedicated parameters.

        self.coordinates = coordinates

        # Intitializing the RNN-----------
        # create either the weight-sharing RNN object or non-weight-sharing RNN object 
        if self.weight_sharing:
            self.DRNN = DilatedRNN_WeightSharing(N,units=self.units,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation=self.activation_function, seed=seed) #contains the graph with the RNNs
        else:
            self.DRNN = DilatedRNN(N,units=self.units,cell=tf.compat.v1.nn.rnn_cell.BasicRNNCell, activation=self.activation_function, seed=seed) #contains the graph with the RNNs

    def run(self):
        """
        vca.run() builds an RNN architecture based on the choice of hyperparameters provided in vca.__init__(). The model then autoregressively generates numsamples (see __init__()) 
        solutions/samples to the optimization problem from which the energies, along with the probabilities of samples and temperature, is used to compute the cost function. 
        Using this cost, the RNN parameters are updated. This process repeats over n_warmup steps with the temperature fixed at T0, and subsequently over n_anneal*n_train steps 
        with the temperature decreasing at every n_anneal step.
        energies_final - energies of 500,000 samples (see line 627) generated after training the model 
        samples_final - samples (or solutions) in the form binary vectors generated after training the model
        """
        #Building the graph -------------------
        with tf.compat.v1.variable_scope(self.DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.DRNN.graph.as_default():

                global_step = tf.Variable(0, trainable=False)
                learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
                learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

                #Defining Tensorflow placeholders
                Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[self.numsamples])
                sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[self.numsamples,self.N])
                log_probs_forgrad = self.DRNN.log_probability(sampleplaceholder_forgrad,inputdim=self.N)
                samplesandprobs = self.DRNN.sample(numsamples=self.numsamples,inputdim=self.N)

                T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

                #Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
                Floc = Eloc + T_placeholder*log_probs_forgrad
                cost = tf.reduce_mean(tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

                gradients, variables = zip(*optimizer.compute_gradients(cost))
                #Calculate Gradients---------------
                #Define the optimization step
                optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)
                #Tensorflow saver to checkpoint
                saver=tf.compat.v1.train.Saver()
                #For initialization
                init=tf.compat.v1.global_variables_initializer()

        #Starting Session------------
        #GPU management
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.compat.v1.Session(graph=self.DRNN.graph, config=config)
        sess.run(init)

        with tf.compat.v1.variable_scope(self.DRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.DRNN.graph.as_default():

                # containers to hold warm up energies
                WU_meanEnergy = []
                WU_varEnergy = []
                samplesandprobs = self.DRNN.sample(numsamples=self.numsamples,inputdim=self.N)
                samples = np.ones((self.numsamples, self.N), dtype=np.int32)

                # counter for time
                time_count = 0
                T = self.T0

                # Warmup Loop
                for it in range(self.num_warmup):
                    
                    # start the timer
                    if it == 0:
                        start = time.time()
                    
                    samples, log_probs = sess.run(samplesandprobs)
                    energies = TSP_energy_2D(self.coordinates, samples)

                    meanE = np.mean(energies)
                    varE = np.var(energies)

                    # append the elements
                    WU_meanEnergy.append(meanE)
                    WU_varEnergy.append(varE)

                    # compute free energy and its variance
                    meanF = np.mean(energies + T*log_probs)
                    varF = np.var(energies + T*log_probs)

                    # Do gradient step
                    sess.run(optstep,feed_dict={Eloc:energies,sampleplaceholder_forgrad:samples,learningrate_placeholder: self.lr, T_placeholder:T})

                    if it%5==0:
                        print('WARM UP PHASE')
                        print('mean(E): {0}, mean(F): {1}, var(E): {2}, var(F): {3}, #samples {4}, #Training step {5}'.format(meanE,meanF,varE,varF,self.numsamples, it))
                        print("Temperature: ", T)        

                    if time_count%5 == 0 and time_count>0:
                        print("Elapsed time is =", time.time()-start, " seconds")
                        print('\n\n')

                    time_count += 1    
            
                # containers for annealing loop
                meanEnergy=[]
                varEnergy=[]
                varFreeEnergy = []
                meanFreeEnergy = []
                temperatures = []
                
                # annealing loop
                for it0 in range(self.num_anneal):

                    # reduce the temperature
                    T = self.T0*(1-it0/self.num_anneal)
                    temperatures.append(T)
                    
                    # training loop
                    for it1 in range(self.num_train):

                        samples, log_probs = sess.run(samplesandprobs)

                        energies = TSP_energy_2D(self.coordinates, samples)

                        meanE = np.mean(energies)
                        varE = np.var(energies)

                        #adding elements to be saved
                        meanEnergy.append(meanE)
                        varEnergy.append(varE)

                        meanF = np.mean(energies + T*log_probs)
                        varF = np.var(energies + T*log_probs)

                        meanFreeEnergy.append(meanF)
                        varFreeEnergy.append(varF)

                        if it1%5==0:
                            print('ANNEALING PHASE')
                            print('mean(E): {0}, mean(F): {1}, var(E): {2}, var(F): {3}, #samples {4}, #Training step {5}'.format(meanE,meanF,varE,varF,self.numsamples, it))
                            print("Temperature: ", T)

                        # Do gradient step
                        sess.run(optstep,feed_dict={Eloc:energies,sampleplaceholder_forgrad:samples,learningrate_placeholder: self.lr, T_placeholder:T})

                        if time_count%5 == 0 and time_count>0:
                            print("Ellapsed time is =", time.time()-start, " seconds")
                            print('\n\n')

                        time_count += 1

                # when training is done, generate 500000 samples, 50000 at a time
                samples_per_step = 1000
                n_steps = 10
                samplesandprobs_final = self.DRNN.sample(numsamples=samples_per_step, inputdim=self.N)

                samples_final = np.ones((samples_per_step*n_steps, self.N), dtype=np.int32)
                energies_final = np.ones((samples_per_step*n_steps))

                for i in range(n_steps):
                    samples_step, _ = sess.run(samplesandprobs_final)
                    energies_step = TSP_energy_2D(self.coordinates, samples_step)
                    samples_final[(i)*samples_per_step : (i+1)*samples_per_step] = samples_step
                    energies_final[(i)*samples_per_step : (i+1)*samples_per_step] = energies_step
                
                print("10,000 samples generated after training")
                    
        return energies_final, samples_final

#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import random
import time

"""
VCA class
"""
# seed for experimental reproducibility
seed = 111
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

def hamiltonian(samples, Q):
    """
    hamiltonian returns the energies of an array of max-cut solutions.
    samples - array of solutions
    Q - QUBO matrix
    energies - returns the energy of the samples
    """
    xQ = np.dot(samples, Q)
    energies = np.einsum('ij,ij->i', xQ, samples)
    return energies

"""
RNNProbability classes
"""
class RNNProbabilityNWS(object):
    def __init__(self,systemsize,cell=None,activation=tf.nn.relu,units=[10],scope='RNNProbability', seed = 111):
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
        self.graph = tf.Graph()
        self.scope = scope #Label of the RNN probability
        self.N = systemsize #Number of sites of the 1D chain

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        # Defining the neural network
        # different RNN block being used at every site
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                self.rnn=[tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[i],activation = activation,name='RNN_{0}{1}'.format(i,n), dtype = tf.float64) for i in range(len(units))]) for n in range(self.N)]
                self.dense = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='PRNN_dense_{0}'.format(n), dtype = tf.float64) for n in range(self.N)]

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

        with self.graph.as_default(): # Call the default graph, used if willing to create multiple graphs.
            samples = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                
                # b = zeros ndarray with shape [numsamples,inputdim] 
                # b = state of one spin for all the samples, this command above makes all the samples having 1 in the first component and 0 in the second.
                b=np.zeros((numsamples,inputdim)).astype(np.float64)

                probs=[]

                # inputs = b ndarray used as a template to create equivalent tensor
                # Initial input to feed to the rnn
                inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.

                self.inputdim=inputs.shape[1]   # 2
                self.outputdim=self.inputdim    # 2
                self.numsamples=inputs.shape[0] # M 

                rnn_state=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)  # [numsamples,num_units]

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn[n](inputs, rnn_state)  # elu activation function output of shape [numsamples,num_units] 
                    output=self.dense[n](rnn_output)    # softmax probability of +1 and -1 spins [numsamples, 2]
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,])   # sampled nth position spin; values are either 0 or 1 [numsamples,]
                    probs.append(output) 
                    samples.append(sample_temp)     # [N, numsamples]
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)     # one-hot encoded vector input for subsequent RNN [numsamples, 2]

            self.samples=tf.stack(values=samples,axis=1) # [self.N, num_samples] to [num_samples, self.N]: Generate self.numsamples vectors of size self.N spin containing 0 or 1

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

class RNNProbabilityWS(object):
    def __init__(self,systemsize,cell=None,activation=tf.nn.relu,units=[10],scope='RNNProbability', seed = 111):
        """
            systemsize:  int
                         number of sites
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            activation:  activation function of the RNN cell
            seed:        pseudo-random number generator
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN probability
        self.N=systemsize #Number of sites of the 1D chain

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                self.rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell(units[i],activation = activation,name='RNN_{0}'.format(i), dtype = tf.float64) for i in range(len(units))])
                self.dense = tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='PRNN_dense', dtype = tf.float64) 

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
            samples = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                b=np.zeros((numsamples,inputdim)).astype(np.float64)
                #b = state of one spin for all the samples, this command above makes all the samples having 1 in the first component and 0 in the second.
                probs=[]

                inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float64)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                    output=self.dense(rnn_output)
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

                rnn_state=self.rnn.zero_state(self.numsamples,dtype=tf.float64)

                for n in range(self.N):
                    rnn_output, rnn_state = self.rnn(inputs, rnn_state)
                    output=self.dense(rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

class RNNProbabilityDilated(object):
    def __init__(self,systemsize,cell=None,activation=tf.nn.relu,units=[2],scope='RNNwavefunction', seed = 111):
        """
            systemsize:  int, size of the lattice
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            Seed         int, pseudo random generate to guarantee reproducibility
        """

        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of nodes
        self.numlayers = len(units)
        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(n)+str(i),dtype=tf.float64) for n in range(self.N)] for i in range(self.numlayers)]
                self.dense = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(n)) for n in range(self.N)] #Define the Fully-Connected layer followed by a Softmax

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
        with self.graph.as_default(): # Call the default graph, used if willing to create multiple graphs.
            samples = []
            probs = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                b=np.zeros((numsamples,inputdim)).astype(np.float64)

                inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

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
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)

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
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

class vca:

    # allows user to decide what type of RNN unit to use among basicRNN, lstmRNN, gru
    rnn_cells = {
        'basic': tf.compat.v1.nn.rnn_cell.BasicRNNCell, 
        'lstm': tf.compat.v1.nn.rnn_cell.LSTMCell,
        'gru': tf.compat.v1.nn.rnn_cell.GRUCell,
        }

    def __init__(self, N, n_layers, n_warmup, n_anneal, n_train, qubo, RNNtype, rnn_unit, T0):
        """
        N - system size of optimization problem
        n_layers - number of RNN layers 
        n_warmup - warmup iterations
        n_anneal - temperature annealing iterations
        n_train - RNN model training iterations
        qubo - qubo matrix of an max-cut graph used to compute the energy of solutions to the max-cut problem
        RNNtype - general architecture of RNN among 
        {'ws': single-chain/VCA-Vanilla with shared parameters at every RNN cell, 
        'nws': single-chain/VCA-Vanilla with dedicated parameters at every RNN cell, 
        'dilated': VCA-Dilated with dedicated parameters}
        rnn_unit - type of RNN cell among {"basicRNN", "GRU", "LSTM"}
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
        self.numlayers = n_layers
        self.num_units = 40     # size of RNN hidden units
        self.units = [self.num_units]*self.numlayers # defines the number of hidden units at each layer (for Dilated)
        self.numsamples = 50    # number of training samples
        self.lr = np.float64(1e-4)  # learning rate for backpropagation
        self.activation_function = tf.nn.elu    # activation function
        self.rnn_cell = vca.rnn_cells[rnn_unit]     # type of RNN cell among {"basic", "lstm", "gru"}

        self.qubo = qubo    # QUBO matrix to calculate the Hamiltonian/energy

        print('\n')
        print("Number of nodes/binary variables = {}".format(self.N))
        print('Number of training samples = {}'.format(self.numsamples))
        print("Initial temperature {}".format(self.T0))

        print('\nWamup steps = {}'.format(self.num_warmup))
        print('Annealing steps = {}'.format(self.num_anneal))
        print('Training steps at a fixed temperature= {}'.format(self.num_train))
        print('Total training steps = {}\n'.format(self.num_warmup+self.num_anneal*self.num_train))

        print('Seed = ', seed)
        print("Number of layers = {0}\n".format(self.numlayers))

        # Intitializing the RNN-----------
        # create either the weight-sharing RNN object or non-weight-sharing RNN object 
        if RNNtype == 'ws':
            self.PRNN = RNNProbabilityWS(self.N, units=self.units, cell=self.rnn_cell, activation=self.activation_function, seed=seed) #contains the graph with the RNNs
        elif RNNtype == 'nws':
            self.PRNN = RNNProbabilityNWS(self.N, units=self.units, cell=self.rnn_cell, activation=self.activation_function, seed=seed)
        elif RNNtype == 'dilated':
            self.PRNN = RNNProbabilityDilated(self.N, units=self.units, cell=self.rnn_cell, activation=self.activation_function, seed=seed)

#Loading previous trainings----------
    ### To be implemented
#------------------------------------
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
        with tf.compat.v1.variable_scope(self.PRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.PRNN.graph.as_default():

                global_step = tf.Variable(0, trainable=False)
                learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
                learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

                E_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[self.numsamples])
                samples_placeholder = tf.compat.v1.placeholder(dtype=tf.int32,shape=[self.numsamples,self.N])
                log_probs_tensor = self.PRNN.log_probability(samples_placeholder,inputdim=2)
                T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

                #Fake Cost function = energy_term + entropy_term
                F = E_placeholder+T_placeholder*log_probs_tensor
                fake_cost = tf.reduce_mean(log_probs_tensor*tf.stop_gradient(F)) - tf.reduce_mean(log_probs_tensor)*tf.reduce_mean(tf.stop_gradient(F)) 
                #fake_cost != F_RNN
                
                gradients, variables = zip(*optimizer.compute_gradients(fake_cost))
                #Calculate Gradients---------------

                optstep = optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)

                init = tf.compat.v1.global_variables_initializer()

        #Starting Session------------
        #GPU management
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.compat.v1.Session(graph=self.PRNN.graph, config=config)
        sess.run(init)

        with tf.compat.v1.variable_scope(self.PRNN.scope,reuse=tf.compat.v1.AUTO_REUSE):
            with self.PRNN.graph.as_default():

                # containers to hold warm up energies
                WU_meanEnergy = []
                WU_varEnergy = []
                

                samplesandprobs = self.PRNN.sample(numsamples=self.numsamples,inputdim=2)
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
                    energies = hamiltonian(samples, self.qubo)

                    meanE = np.mean(energies)
                    varE = np.var(energies)

                    # append the elements
                    WU_meanEnergy.append(meanE)
                    WU_varEnergy.append(varE)

                    # compute free energy and its variance
                    meanF = np.mean(energies + T*log_probs)
                    varF = np.var(energies + T*log_probs)

                    # Do gradient step
                    sess.run(optstep,feed_dict={E_placeholder:energies,samples_placeholder:samples,learningrate_placeholder: self.lr, T_placeholder:T})

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

                        energies = hamiltonian(samples, self.qubo)

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
                        sess.run(optstep,feed_dict={E_placeholder:energies,samples_placeholder:samples,learningrate_placeholder: self.lr, T_placeholder:T})

                        if time_count%5 == 0 and time_count>0:
                            print("Ellapsed time is =", time.time()-start, " seconds")
                            print('\n\n')

                        time_count += 1

                # when training is done, generate 500000 samples, 50000 at a time
                samples_per_step = 1000
                n_steps = 10
                samplesandprobs_final = self.PRNN.sample(numsamples=samples_per_step, inputdim=2)

                samples_final = np.ones((samples_per_step*n_steps, self.N), dtype=np.int32)
                energies_final = np.ones((samples_per_step*n_steps))

                for i in range(n_steps):
                    samples_step, _ = sess.run(samplesandprobs_final)
                    energies_step = hamiltonian(samples_step, self.qubo)
                    samples_final[(i)*samples_per_step : (i+1)*samples_per_step] = samples_step
                    energies_final[(i)*samples_per_step : (i+1)*samples_per_step] = energies_step
                
                print("10,000 samples generated after training")
                    
        return energies_final, samples_final

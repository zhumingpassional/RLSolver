import numpy as np

"""
This implementation is based on RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Description: we define helper functions to obtain the local energies of a 1D model 
both with and without the presence of a transverse magnetic field
"""

def Ising_diagonal_matrixelements(Jz, samples):
        """ To get the diagonal matrix elements of 1D Ising chain given a set of set of samples in parallel!
        Returns: The local energies that correspond to the "samples"
        Inputs:
        - samples: (numsamples, N)
        - Jz: (N-1) np array
        """
        numsamples = samples.shape[0]
        N = samples.shape[1]
        energies = np.zeros((numsamples), dtype = np.float64)

        for i in range(N-1): #diagonal elements
            values = samples[:,i]+samples[:,i+1]
            valuesT = np.copy(values)
            valuesT[values==2] = +1
            valuesT[values==0] = +1
            valuesT[values==1] = -1

            energies += valuesT*(-Jz[i])

        return energies

def Ising_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
        """ To get the local energies of 1D spin chain given a set of set of samples in parallel!
        Returns: The local energies that correspond to the "samples"
        Inputs:
        - samples: (numsamples, N)
        - Jz: (N-1) np array
        - Bx: float
        - queue_samples: ((N+1)*numsamples, N) an empty allocated np array to store all the sample applying the Hamiltonian H on samples
        - log_probs_tensor: A TF tensor with size (None)
        - samples_placeholder: A TF placeholder to feed in a set of configurations
        - log_probs: ((N+1)*numsamples): an allocated np array to store the log_probs non diagonal elements
        - sess: The current TF session
        """
        numsamples = samples.shape[0]
        N = samples.shape[1]

        local_energies = np.zeros((numsamples), dtype = np.float64)

        for i in range(N-1): #diagonal elements (let's do 1D for simple stuff)
            values = samples[:,i]+samples[:,i+1]
            valuesT = np.copy(values)
            valuesT[values==2] = +1 #If both spins are up
            valuesT[values==0] = +1 #If both spins are down
            valuesT[values==1] = -1 #If they are opposite

            local_energies += valuesT*(-Jz[i])

        queue_samples[0] = samples #storing the diagonal samples

        if Bx != 0:
            for i in range(N):  #Non-diagonal elements
                valuesT = np.copy(samples)
                valuesT[:,i][samples[:,i]==1] = 0 #Flip spin i
                valuesT[:,i][samples[:,i]==0] = 1 #Flip spin i

                queue_samples[i+1] = valuesT

            len_sigmas = (N+1)*numsamples
            steps = len_sigmas//50000+1 #I want a maximum of 50000 in batch size just to be safe I don't allocate too much memory

            queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
            for i in range(steps):
              if i < steps-1:
                  cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
              else:
                  cut = slice((i*len_sigmas)//steps,len_sigmas)

              # Compute the log-probabilities with Tensorflow
              log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})

            log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])


            for j in range(numsamples):
                local_energies[j] += -Bx*np.sum( np.exp(0.5*log_probs_reshaped[1:,j]   - 0.5*log_probs_reshaped[0,j])  )

        return local_energies

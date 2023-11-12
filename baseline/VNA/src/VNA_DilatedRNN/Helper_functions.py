import numpy as np
import tensorflow as tf

"""
Code by Mohamed Hibat-Allah
Description: we define helper functions to obtain the local energies of a fully connected model 
both with and without the presence of a transverse magnetic field
"""

# Loading Functions --------------------------
def Fullyconnected_diagonal_matrixelements(Jz, samples):
    """ To get the diagonal local energies of a fully-connected spin model given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, N)
    - Jz: (N,N) np array of J_ij couplings
    """

    numsamples = samples.shape[0]
    N = samples.shape[1]
    energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1):
      values = np.expand_dims(samples[:,i], axis = -1)+samples[:,i+1:]
      valuesT = np.copy(values)
      valuesT[values==2] = +1 #If both spins are up
      valuesT[values==0] = +1 #If both spins are down
      valuesT[values==1] = -1 #If they are opposite

      energies += np.sum(valuesT*(-Jz[i,i+1:]), axis = 1)

    return energies

def Fullyconnected_localenergies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ To get the local energies of a fully-connected spin model given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, N)
    - Jz: (N,N) np array of J_ij couplings
    - Bx: float
    - queue_samples: ((N+1)*numsamples, N) an allocated np array to store all the sample applying the Hamiltonian H on samples
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((N+1)*numsamples): an allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """

    numsamples = samples.shape[0]
    N = samples.shape[1]
    
    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1):
      # for j in range(i+1,N):
      values = np.expand_dims(samples[:,i], axis = -1)+samples[:,i+1:]
      valuesT = np.copy(values)
      valuesT[values==2] = +1 #If both spins are up
      valuesT[values==0] = +1 #If both spins are down
      valuesT[values==1] = -1 #If they are opposite

      local_energies += np.sum(valuesT*(-Jz[i,i+1:]), axis = 1)
    # local_energies += -N*np.mean((2*samples-1), axis = 1)**3

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        count = 0
        for i in range(N-1):  #Non-diagonal elements
            valuesT = np.copy(samples)
            valuesT[:,i][samples[:,i]==1] = 0 #Flip spin i
            valuesT[:,i][samples[:,i]==0] = 1 #Flip spin i


            count += 1
            queue_samples[count] = valuesT

        len_sigmas = (N+1)*numsamples
        steps = len_sigmas//50000+1 #I want a maximum of 50000 in batch size just to be safe I don't allocate too much memory

        queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, N])
        for i in range(steps):
          if i < steps-1:
              cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
          else:
              cut = slice((i*len_sigmas)//steps,len_sigmas)
          log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})


        log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
        for j in range(numsamples):
            local_energies[j] += -Bx*np.sum(0.5*(np.exp(log_probs_reshaped[1:,j]-log_probs_reshaped[0,j])))

    return local_energies

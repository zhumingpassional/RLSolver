import numpy as np
import tensorflow as tf

"""
This implementation is based on RNN Wave Functions' code https://github.com/mhibatallah/RNNWavefunctions
Edited by Mohamed Hibat-Allah
Description: we define helper functions to obtain the local energies of a 2D model 
both with and without the presence of a transverse magnetic field
"""

# Loading Functions --------------------------
def Ising2D_diagonal_matrixelements(Jz, samples):
    """ To get the diagonal local energies of 2D spin lattice given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny,2) np array of J_ij couplings
    """

    numsamples = samples.shape[0]
    Nx = samples.shape[1]
    Ny = samples.shape[2]

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:,0]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice :)))
        values = samples[:,:,i]+samples[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i,1]), axis = 1)

    return local_energies
#--------------------------

def Ising2D_local_energies(Jz, Bx, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs, sess):
    """ To get the local energies of 2D spin lattice given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny,2) np array of J_ij couplings
    - Bx: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx,Ny) an allocated np array to store all the sample applying the Hamiltonian H on samples
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((Nx*Ny+1)*numsamples): an allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """

    numsamples = samples.shape[0]
    Nx = samples.shape[1]
    Ny = samples.shape[2]

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[i,:,0]), axis = 1)

    for i in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice :)))
        values = samples[:,:,i]+samples[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite

        local_energies += np.sum(valuesT*(-Jz[:,i,1]), axis = 1)


    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        for i in range(Nx):  #Non-diagonal elements
            for j in range(Ny):
                valuesT = np.copy(samples)
                valuesT[:,i,j][samples[:,i,j]==1] = 0 #Flip
                valuesT[:,i,j][samples[:,i,j]==0] = 1 #Flip

                queue_samples[i*Ny+j+1] = valuesT

        len_sigmas = (N+1)*numsamples
        steps = len_sigmas//50000+1 #I want a maximum in batch size just to not allocate too much memory
        # print("Total num of steps =", steps)
        queue_samples_reshaped = np.reshape(queue_samples, [(N+1)*numsamples, Nx,Ny])
        for i in range(steps):
          if i < steps-1:
              cut = slice((i*len_sigmas)//steps,((i+1)*len_sigmas)//steps)
          else:
              cut = slice((i*len_sigmas)//steps,len_sigmas)
          log_probs[cut] = sess.run(log_probs_tensor, feed_dict={samples_placeholder:queue_samples_reshaped[cut]})
          # print(i)

        log_probs_reshaped = np.reshape(log_probs, [N+1,numsamples])
        for j in range(numsamples):
            local_energies[j] += -Bx*np.sum(np.exp(0.5*log_probs_reshaped[1:,j]-0.5*log_probs_reshaped[0,j]))

    return local_energies

import numpy as np
import os

"""
Functions
"""
def read_cities(path):
    """
    Read the x and y coordinates of the list of cities from a .txt/.SPARSE file.
    path - path of the file
    coordinates_X - x coordinates of the cities
    coordinates_Y - y coordinates of the cities
    """
    with open(path, 'r') as f:
        N = int(f.readline())
        coordinates_X = []
        coordinates_Y = []
        for _ in range(N):
            line = f.readline().split()
            x, y = float(line[0]), float(line[1])
            coordinates_X.append(x)
            coordinates_Y.append(y)

    return coordinates_X, coordinates_Y, N

def energy(samples_X, samples_Y):
    """
    Compute TSP energies of samples i.e. the summed Euclidian distance.
    samples_X - x coordinates of samples
    samples_Y - y coordinates of samples
    energies - energies of the samples 
    """
    M = samples_X.shape[0]
    N = samples_X.shape[1]
    energies = np.zeros(M, dtype=np.float64)
    for i in range(N):
        energies += np.sqrt(np.power(samples_X[:,(i+1)%N] - samples_X[:,i], 2) + np.power(samples_Y[:,(i+1)%N] - samples_Y[:,i], 2))

    return energies

"""
SA class
"""
class SA:
    """
    Class for solving the travelling salesman problem (TSP) using Simulated Annealing.
    The TSP input is a set of Cartesian coordinates: points_X, points_Y. The cost function 
    is given by adding up the individual Euclidiean distances of every two consecutvie 
    points on the tour. This implementation generates M independent solutions.
    || HOW TO USE ||
    create an SA object --> solver = SA(...)
    solve to get TSP routes and the corresponsing energies --> samples_X, samples_Y, energies = solver.solve()
    """
    # seed for data reproducibility
    seed = 111
    np.random.seed(seed)

    # a more optimized way of computing the energy of new samples as opposed to computing the energy using energy() function 
    # note that this only works as long as information of the current samples' energies are available
    def energy_change(samples_X, samples_Y, samples_X_new, samples_Y_new, indices1, indices2):
        """
        Compute the energy change between current samples and new samples.
        Take into account only the connections removed (in current samples) and the ones constructued (in new samples).
        samples_X - x coordinates of current samples 
        samples_Y - y coordinates of current samples
        samples_X_new - x coordinates of new samples 
        samples_Y_new - y coordinates of new samples
        indices1 - indices of the first set of cities swapped
        indices2 - indices of the second set of cities swapped
        delta - energy change between current samples and new samples for all samples
        """
        M = samples_X.shape[0]
        N = samples_X.shape[1]
        delta = np.zeros(M)
        # remove energy connections from current samples and reduce delta
        delta -= np.sqrt((samples_X[np.arange(M), indices1] - samples_X[np.arange(M), (indices1-1)%N])**2 + (samples_Y[np.arange(M), indices1] - samples_Y[np.arange(M), (indices1-1)%N])**2)
        delta -= np.sqrt((samples_X[np.arange(M), indices1] - samples_X[np.arange(M), (indices1+1)%N])**2 + (samples_Y[np.arange(M), indices1] - samples_Y[np.arange(M), (indices1+1)%N])**2)
        delta -= np.sqrt((samples_X[np.arange(M), indices2] - samples_X[np.arange(M), (indices2-1)%N])**2 + (samples_Y[np.arange(M), indices2] - samples_Y[np.arange(M), (indices2-1)%N])**2)
        delta -= np.sqrt((samples_X[np.arange(M), indices2] - samples_X[np.arange(M), (indices2+1)%N])**2 + (samples_Y[np.arange(M), indices2] - samples_Y[np.arange(M), (indices2+1)%N])**2)
        # construct energy connections for new samples and increase delta
        delta += np.sqrt((samples_X_new[np.arange(M), indices1] - samples_X_new[np.arange(M), (indices1-1)%N])**2 + (samples_Y_new[np.arange(M), indices1] - samples_Y_new[np.arange(M), (indices1-1)%N])**2)
        delta += np.sqrt((samples_X_new[np.arange(M), indices1] - samples_X_new[np.arange(M), (indices1+1)%N])**2 + (samples_Y_new[np.arange(M), indices1] - samples_Y_new[np.arange(M), (indices1+1)%N])**2)
        delta += np.sqrt((samples_X_new[np.arange(M), indices2] - samples_X_new[np.arange(M), (indices2-1)%N])**2 + (samples_Y_new[np.arange(M), indices2] - samples_Y_new[np.arange(M), (indices2-1)%N])**2)
        delta += np.sqrt((samples_X_new[np.arange(M), indices2] - samples_X_new[np.arange(M), (indices2+1)%N])**2 + (samples_Y_new[np.arange(M), indices2] - samples_Y_new[np.arange(M), (indices2+1)%N])**2)
        
        return delta

    def initialize_samples(M, points_X, points_Y):
        """
        Generate M number of random permutations of points_X and points_Y in unison.
        M - number of samples to generate
        points_X - input x coordinates
        points_Y - input y coordinates
        return samples_X - M number of shuffled points_X
        return samples_Y - M number of shuffled points_Y
        """
        # create 2D array of repeated points_X/points_Y vectors
        samples_X = np.array([points_X for _ in range(M)])
        samples_Y = np.array([points_Y for _ in range(M)])
        # generate M number of random indices from 0 --> N-1
        # then index samples_X and samples_Y in unison according to these indices
        shuffled_indices = np.array([np.random.permutation(len(points_X)) for _ in range(M)])
        samples_X = samples_X[[[i] for i in range(M)], shuffled_indices]
        samples_Y = samples_Y[[[i] for i in range(M)], shuffled_indices]

        return samples_X, samples_Y

    def __init__(self, N, points_X, points_Y, M=50, n_warmup=2000, n_anneal=16, n_eq=5, T0=2.0):
        """
        N - system size or the number of cities in the TSP configuration
        points_X - input x coordinates
        points_Y - input y coordinates
        M - number of SA solutions to be processed independently
        n_warmup - number of warm-up steps
        n_anneal - number of annealing steps
        n_eq - number of equilibrium steps
        T0 - initial temperature in the annealing schedule
        """
        # ensure list of x, y coordinates are the same length
        assert len(points_X) == len(points_Y)
        assert(N == len(points_X))
        self.N = N
        self.M = M
        # initialize samples and get their enrgy
        self.samples_X, self.samples_Y = SA.initialize_samples(self.M, points_X, points_Y)
        self.energies = energy(self.samples_X, self.samples_Y)
        # SA parameters
        self.n_warmup = n_warmup
        self.n_anneal = n_anneal
        self.n_eq = n_eq
        self.T0 = T0

    def solve(self):
        """
        Method that solves the input TSP configuration and generates M solution.
        samples_X - x coordinates of the solutions
        samples_Y - y coordinates of the solutions
        energies - energies of the the solutions
        """
        # warm-up process
        for _ in range(self.n_warmup):
            # sweep
            for _ in range(self.N):
                # Metropolis-Hastings step
                self.metropolis(self.T0)

        # annealing procss
        for i in range(self.n_anneal):
            # compute temperature T
            T = self.T0 - self.T0*i/self.n_anneal
            # equilibriate at current T
            for _ in range(self.n_eq):
                # sweep
                for _ in range(self.N):
                    self.metropolis(T)

        return self.samples_X, self.samples_Y, self.energies
    
    def metropolis(self, T):
        """
        Metropolis-Hastings step that generates a new solution by swapping cities at two positions in the current solution vector.
        If the new sample has a lower energy than the current one, the former replaces the latter. Otherwise, the new solution may still replace the 
        current one based on some probability. This is done concurrently for all M solutions.
        T - temperature at which to perform the Metropolis-Hastings step
        """
        # generate two sets of random indices for all samples
        indices1 = np.random.randint(0, self.N, size=self.M)
        indices2 = np.random.randint(0, self.N, size=self.M)
        # swap X, Y index values in unison for all samples
        samples_X_new, samples_Y_new = self.samples_X.copy(), self.samples_Y.copy()
        samples_X_new[np.arange(self.M), indices1], samples_X_new[np.arange(self.M), indices2] = samples_X_new[np.arange(self.M), indices2], samples_X_new[np.arange(self.M), indices1]
        samples_Y_new[np.arange(self.M), indices1], samples_Y_new[np.arange(self.M), indices2] = samples_Y_new[np.arange(self.M), indices2], samples_Y_new[np.arange(self.M), indices1]
        # find delta of enregies and compute the probabilities of new samples replacing current ones
        delta = SA.energy_change(self.samples_X, self.samples_Y, samples_X_new, samples_Y_new, indices1, indices2)
        energies_new = self.energies + delta
        probabilities = np.exp(-delta/T)
        # probabilistically obtain a boolean array where True means successful replacement of new samples. False means keeping the current samples
        replacement_indices = probabilities > np.random.random(size=self.M)
        # replace current samples based on replacement_indices
        self.samples_X[replacement_indices] = samples_X_new[replacement_indices]
        self.samples_Y[replacement_indices] = samples_Y_new[replacement_indices]
        self.energies[replacement_indices] = energies_new[replacement_indices]

"""
Main
"""
if __name__ == '__main__':
    # set the appropriate filepath to the Max-Cut graph below
    filepath = "../Data/TSP Instances/coordinates_N64.txt"
    assert(os.path.exists(filepath)) # check if file exists in path
    coordinates_X, coordinates_Y, N = read_cities(filepath)
    # run the SA solver
    sa_solver = SA(N=N, points_X=coordinates_X, points_Y=coordinates_Y, M=50, n_warmup=100, n_anneal=16, n_eq=5, T0=2.0)
    samples_coordX, samples_coordY, energies = sa_solver.solve()
    samples = [list(zip(samples_coordX[i], samples_coordY[i])) for i in range(len(samples_coordX))] # samples contains M instances of [(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)]


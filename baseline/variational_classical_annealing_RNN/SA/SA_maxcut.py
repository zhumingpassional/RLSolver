import numpy as np
import random as rd
import os

"""
Functions
"""
def read_graph(path):
    """
    Read the Max-Cut graph stored in a .txt/.SPARSE file and returns an adjacency matrix of the Max-Cut graph.
    path - filepath of the Max-Cut graph file
    adj_matrix - returns adjacency matrix of Max-Cut instance
    """
    with open(path) as f:
        line = f.readline().split()
        # get number of nodes and number of undirected edges
        N, entries = int(line[0]), int(line[1])
        # create a zeros QUBO matrix
        adj_matrix = np.zeros(shape=(N,N))
        for _ in range(entries):
            # extract the node indices and value; fill the adjacency matrix
            line = f.readline().split()
            node1, node2, value = int(line[0]), int(line[1]), int(line[2])
            # fill both Q[i,j] and Q[j,i] as the QUBO matrix is symmetric
            adj_matrix[node1-1, node2-1] = value
            adj_matrix[node2-1, node1-1] = value

        return adj_matrix, N

def formulate_qubo(adj_matrix):
    """
    Take an adjacency matrix of a Max-Cut instance as parameter and generate the QUBO matrix of that instance.
    adj_matrix - adjacency matrix of Max-Cut instance
    qubo - QUBO matrix of Max-Cut instance
    """    
    qubo = adj_matrix.copy()
    # get the number of edges of each node by summing the respective row in G
    edge_count = np.sum(qubo, axis=1)
    # fill the diagonal positions with the corresponding edge_count * -1
    np.fill_diagonal(qubo, -edge_count)
    return qubo

def hamiltonian(samples, Q):
    """
    hamiltonian returns the energies of an array of max-cut solutions.
    samples - array of solutions
    Q - QUBO matrix
    energies - returns the energy of the samples
    """
    xQ = np.dot(samples, Q)
    energies = np.einsum('ij,ij->i', xQ, samples)   # numpy einsum computation
    return energies

"""
SA class
"""
class SA:
    """
    Class for solving the maximum cut problem (Max-Cut) using Simulated Annealing.
    The solver primary requires the QUBO matrix of the Max-Cut instance which is used to
    compute the energy of a solution. The implementation generates n_samples number of 
    independent solutions.
    || HOW TO USE ||
    create an SA object --> solver = SA(...)
    solve to get the Max-Cut solutions (in binary vectors) and the corresponsing energies --> samples, energies = solver.solve()
    """
    seed = 111
    rd.seed(seed)
    np.random.seed(seed)

    def __init__(self, N, qubo, n_samples, n_warmup, n_anneal, n_eq, T0=2.0):
        """
        N - N - system size or the number of vertices in the max-cut graph
        qubo - qubo matrix of Max-Cut instance
        n_samples - number of independent samples that are concurrently processed
        n_warmup - there are N metropolis moves per warmup step at temperature T=T0
        n_anneal - number of annealing steps where temperature is decreased after which the equilibrating steps follows
        n_eq - number of equilibrating steps after which there are N metropolis steps
        T0 - initial temperature
        """
        # check if the provided system size matches with the qubo matrix
        assert(N == qubo.shape[0])
        # size of the system
        self.n_samples = n_samples
        self.N = N
        # warm up and annealing loops
        self.T0 = T0
        self.n_warmup = n_warmup
        self.n_anneal = n_anneal
        self.n_eq = n_eq
        self.qubo = qubo

        # initialize random binary vector and its E
        self.samples = np.random.choice([0,1], size=(self.n_samples, self.N))
        self.energies = hamiltonian(self.samples, self.qubo)

    def solve(self):
        """
        Method solves max-cut instance with the simulated annealing technique.
        samples - binary vector solutions to the max-cut problem
        energies - corresponding energy value of the solutions
        """
        # warm up process
        for _ in range(self.n_warmup):
            # Sweep
            for _ in range(self.N):
                # metropolis step
                self.metropolis(self.T0)

        # annealing step
        for i in range(self.n_anneal):
            T = self.T0 - self.T0*i/self.n_anneal
            # equilibrating process
            for _ in range(self.n_eq):
                # sweep
                for _ in range(self.N):
                    # metropolis step
                    self.metropolis(T)

        return self.samples, self.energies


    def metropolis(self, T):
        """
        Metropolis-Hastings step that generates a neighboring solution of the current one by randomly flipping the bit of an element in the solution vector.
        If the new solution has a lower energy than the current one, then the former replaces the latter. Otherwise, the new solution may still replace the 
        current one based on some probability. This is done concurrently for all n_samples solutions.
        T - temperature at which to perform the Metropolis-Hastings step
        """
        # generate a new, neighboring vector and compute its energy
        index = np.random.choice([0,self.N-1], size=self.n_samples)
        index = np.random.randint(0,self.N, size=self.n_samples)
        samples_new = self.samples.copy()
        
        # get the respective binary values
        values = samples_new[np.arange(self.n_samples),index]
        # flip values
        values = np.bitwise_xor(values, 1)
        # create the new samples
        samples_new[np.arange(self.n_samples),index] = values
        
        # compute E of new vector
        energies_new = hamiltonian(samples_new, self.qubo)
        # get the delta between E and E_new
        delta_energies = energies_new - self.energies
        
        # get probs of new samples replacing old ones
        probs = np.exp(-delta_energies/T)
        replace_index = probs > np.random.random(self.n_samples)
        # replace old samples
        self.samples[replace_index] = samples_new[replace_index]
        self.energies[replace_index] = energies_new[replace_index]


"""
Main
"""
if __name__ == '__main__':
    # set the appropriate filepath to the Max-Cut graph below
    filepath = "../Data/maxcut_instances/rudy_128_12_1340.txt"
    assert(os.path.exists(filepath)) # check if file exists in path
    G, N = read_graph(filepath)
    Q = formulate_qubo(G)
    # run the SA solver
    sa_solver = SA(N=N, qubo=Q, n_samples=50, n_warmup=100, n_anneal=16, n_eq=5, T0=2.0)
    samples, energies = sa_solver.solve()


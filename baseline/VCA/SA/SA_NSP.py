import numpy as np
import random as rd

"""
Functions
"""
def dwave_nsp_qubo(n_nurses, n_days):
    """
    Given the number of nurses and days in the NSP instance, formulate the QUBO matrix which can be used to compute 
    the energy of a solution to this problem.
    n_nurses - number of nurses in the NSP configuration
    n_days - number of days in the NSP configuration
    Note: This code in dwave_nsp_qubo is borrowed from the repository: 
    https://github.com/dwave-examples/nurse-scheduling/blob/master/nurse_scheduling.py
    """
    # row, col size of Q matrix
    size = n_nurses * n_days
    # dictionary for mapping (nurse,day) tuple to the index in the solution vector (mapping matrix indices to vector index)
    d = {(nurse,day):nurse*n_days+day for nurse in range(n_nurses) for day in range(n_days)}
    # HNC correlation coeff
    a = 3.5
    # HSC coeff and constants
    hsc_coeff = 1.3
    workload = 1
    effort = 1
    # SNC coeff and constants
    snc_coeff = .3
    min_duty_days = int(n_days/n_nurses)
    preference = 1

    # qubo matrix
    Q = np.zeros(shape=(size,size))
    # placing the HNC entries in Q 
    for nurse in range(n_nurses):
        for day in range(n_days-1):
            i, j = d[(nurse,day)], d[(nurse,day+1)]
            Q[i][j] = a

    # incorporating Hard Shift Constraint penalty to Q 
    for nurse in range(n_nurses):
        for day in range(n_days):
            index = d[(nurse, day)]
            Q[index, index] += hsc_coeff * (effort**2 - (2*workload*effort))

    for day in range(n_days):
        for nurse1 in range(n_nurses):
            for nurse2 in range(nurse1 + 1, n_nurses):
                index1 = d[(nurse1, day)]
                index2 = d[(nurse2, day)]
                Q[index1, index2] += 2*hsc_coeff*effort**2

    # incorporating Soft Nurse Constraint penalty to Q 
    for nurse in range(n_nurses):
        for day in range(n_days):
            index = d[(nurse, day)]
            Q[index, index] += snc_coeff * (preference**2 - (2*preference*min_duty_days))

    for nurse in range(n_nurses):
        for day1 in range(n_days):
            for day2 in range(day1 + 1, n_days):
                index1 = d[(nurse, day1)]
                index2 = d[(nurse, day2)]
                Q[index1, index2] += 2*snc_coeff*preference**2

    offset = hsc_coeff*n_days*workload**2 + snc_coeff*n_nurses*min_duty_days**2
    
    return Q, offset

def hamiltonian(samples, Q, offset):
    """
    hamiltonian returns the energies of an array of NSP solutions.
    samples - array of solutions
    Q - QUBO matrix
    energies - returns the energy of the samples
    """
    xQ = np.dot(samples, Q)
    energies = np.einsum('ij,ij->i', xQ, samples)
    energies += offset
    return energies

"""
SA class
"""
class SA:
    """
    Class for solving the nurse scheduling problem (NSP) using Simulated Annealing.
    The solver primary requires the QUBO matrix of the NSP configuration which is used to
    compute the energy of a solution. The implementation generates n_samples number of 
    separate solutions.
    || HOW TO USE ||
    create an SA object --> solver = SA(...)
    solve to get the NSP solutions (in binary vectors) and the corresponsing energies --> samples, energies = solver.solve()
    """
    # seed for reproducibility
    seed = 111
    rd.seed(seed)
    np.random.seed(seed)

    def __init__(self, N, qubo, offset, n_samples, n_warmup, n_anneal, n_eq, T0=2.0):
        """
        N - system size
        qubo - qubo matrix of NSP instance
        n_samples - number of independent samples that are concurrently processed
        n_warmup - there are N metropolis moves per warmup step at temperature T=T0
        n_anneal - number of annealing steps where temperature is decreased after which the equilibrating steps follows
        n_eq - number of equilibrating steps after which there are N metropolis steps
        offset - constant that is used in energy calculations
        T0 = initial temperature
        """
        self.num_sample = n_samples
        self.N = N
        # warm up and annealing loops
        self.T0 = T0
        self.n_warmup = n_warmup
        self.n_anneal = n_anneal
        self.n_equilibrium = n_eq
        # qubo and offset for energy calculation
        self.qubo = qubo
        self.offset = offset

        # initialize random binary vector and its E
        self.samples = np.random.choice([0,1], size=(self.num_sample, self.N))
        self.energies = hamiltonian(self.samples, self.qubo, self.offset)

    def solve(self):
        """
        Method solves NSP configurations using the simulated annealing technique.
        samples - binary vector solutions to the NSP problem
        energies - corresponding energy value of the solutions
        """
        # warm up process
        for i in range(self.n_warmup):
            # Sweep step
            for j in range(self.N):
                # metropolis step
                self.metropolis(self.T0)
        # annealing step
        for i in range(self.n_anneal):
            T = self.T0 - self.T0*i/self.n_anneal
            # equilibrating step
            for j in range(self.n_equilibrium):
                # sweep
                for k in range(self.N):
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
        # generate a new, neighboring vector and its energy
        index = np.random.choice([0,self.N-1], size=self.num_sample)
        index = np.random.randint(0,self.N, size=self.num_sample)
        samples_new = self.samples.copy()
        
        # get the respective binary vlues
        values = samples_new[np.arange(self.num_sample),index]
        # flip values
        values = np.bitwise_xor(values, 1)
        # create the new samples
        samples_new[np.arange(self.num_sample),index] = values
        
        # compute E of new vector
        energies_new = hamiltonian(samples_new, self.qubo, self.offset)
        # get the delta between E and E_new
        delta_energies = energies_new - self.energies
        
        # get probs of new samples replacing old ones
        probs = np.exp(-delta_energies/T)
        replace_index = probs > np.random.random(self.num_sample)
        # replace old samples
        self.samples[replace_index] = samples_new[replace_index]
        self.energies[replace_index] = energies_new[replace_index]

"""
Main
"""
if __name__ == '__main__':
    # NSP configuration
    D = 15
    N = 7
    system_size = N*D
    Q, offset = dwave_nsp_qubo(N, D)
    sa_solver = SA(N=system_size, qubo=Q, offset=offset, n_samples=50, n_warmup=100, n_anneal=16, n_eq=5, T0=2.0)
    samples, energies = sa_solver.solve()
    np.save("test", energies)

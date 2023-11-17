from vca_nsp import vca
import numpy as np
import time
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

def main():
    """
    Main
    """
    start_time = time.time()
    # obtain the qubo matrix and offset scalar of the NSP configuration
    qubo, offset = dwave_nsp_qubo(n_days=15, n_nurses=7)
    N = qubo.shape[0]  # system size

    # VCA parameters
    # n_warmup - number of training steps at the initial temperature T=T0
    # n_anneal - duration of the annealing procedure
    # n_train - number of training step during backprop after every annealing step
    # dilated_layers - number of dilated layers for VCA-Dilated. Note: for other architectures, layers = 1
    n_warmup = 2000
    n_anneal = 16
    n_train = 5
    dilated_layers = np.int32(np.ceil(np.log2(N)))

    # RNNtype specifies the RNN architecture among {'ws', 'nws', 'dilated'} where
    # 'ws' - weight-sharing RNN parameters
    # 'nws' - independent RNN parameters at every RNN cell
    # 'dilated' - independent RNN parameters at every RNN cell with a dilatedRNN structure
    # rnn_unit specifies the RNN cell type among {'basic', 'lstm', 'gru'}

    # VCA-Dilated
    model = vca(N=N, n_layers=dilated_layers, n_warmup=n_warmup, n_anneal=n_anneal, n_train=n_train, qubo=qubo,
                offset=offset, RNNtype='dilated', rnn_unit='basic', T0=2)
    energies_dilated, samples_dilated = model.run()  # returns numpy nd arrays

    # VCA-Vanilla
    model = vca(N=N, n_layers=1, n_warmup=n_warmup, n_anneal=n_anneal, n_train=n_train, qubo=qubo, offset=offset,
                RNNtype='nws', rnn_unit='basic', T0=2)
    energies_vanilla, samples_vanilla = model.run()  # returns numpy nd arrays
    print("running_duration: %.2f", running_duration)
if __name__ == '__main__':
    main()
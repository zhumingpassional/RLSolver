from vca_maxcut import vca
import numpy as np
import os
import time

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

def main():
    """
    Main
    """
    start_time = time.time()
    # set the filepath of the Max-Cut graph instance below
    # filepath = "../../Data/maxcut_instances/rudy_128_12_1340.txt"
    filepath = "../../../../data/gset/gset_14.txt"
    assert (os.path.exists(filepath))  # check if file exists in path
    graph, N = read_graph(filepath)
    qubo = formulate_qubo(graph)
    # VCA hyperparameters
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
    # rnn_unit specifies the RNN cell type among {'basic', 'lstm', 'gru'} specifies the RNN cell type among {'basic', 'lstm', 'gru'}

    # VCA-Dilated
    model = vca(N=N, n_layers=dilated_layers, n_warmup=n_warmup, n_anneal=n_anneal, n_train=n_train, qubo=qubo,
                RNNtype='dilated', rnn_unit='basic', T0=2.0)
    energies_dilated, samples_dilated = model.run()

    # VCA-Vanilla
    model = vca(N=N, n_layers=1, n_warmup=n_warmup, n_anneal=n_anneal, n_train=n_train, qubo=qubo, RNNtype='nws',
                rnn_unit='basic', T0=2.0)
    energies_vanilla, samples_vanilla = model.run()
    running_duration = time.time() - start_time
    print("running_duration: %.2f", running_duration)

if __name__ == '__main__':
    main()
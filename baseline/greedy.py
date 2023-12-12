# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
import copy
import time
from typing import List, Union
import numpy as np
from typing import List
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut, obj_graph_partitioning
from util import write_result
from util import plot_fig
from util import transfer_nxgraph_to_weightmatrix
from config import *

def greedy_maxcut(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        print("iteration in greedy: ", iteration)
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        for node in nodes:
            new_solution = copy.deepcopy(curr_solution)
            # search a new solution and calc obj
            new_solution[node] = (new_solution[node] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score >= curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_graph_partitioning(init_solution: Union[List[int], np.array], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_graph_partitioning(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_nodes):
        node1 = nodes[i]
        traversal_scores = []
        traversal_solutions = []
        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            new_solution = copy.deepcopy(curr_solution)
            tmp = new_solution[node1]
            new_solution[node1] = new_solution[node2]
            new_solution[node2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        if len(traversal_scores) == 0:
            continue
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score >= curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # read data
    graph = read_nxgraph('../data/syn/syn_50_176.txt')
    weightmatrix = transfer_nxgraph_to_weightmatrix(graph)
    # run alg
    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    num_steps = 30
    alg_name = 'GR'

    # maxcut
    if PROBLEM_NAME == ProblemName.maxcut:
        gr_score, gr_solution, gr_scores = greedy_maxcut(init_solution, num_steps, graph)

    # graph_partitioning
    if PROBLEM_NAME == ProblemName.graph_partitioning:
        gr_score, gr_solution, gr_scores = greedy_graph_partitioning(init_solution, graph)
    # write result
    write_result(gr_solution, '../result/result.txt')
    obj = obj_maxcut(gr_solution, graph)
    print('obj: ', obj)
    
    # plot fig
    plot_fig(gr_scores, alg_name)





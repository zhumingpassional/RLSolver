import copy
import time
from typing import List, Union
import numpy as np
from typing import List
import random
import networkx as nx
from util import read_nxgraph
from util import obj_maxcut, obj_graph_partitioning, obj_minimum_vertex_cover, cover_all_edges
from util import write_result
from util import plot_fig
from config import *
def simulated_annealing(init_solution: Union[List[int], np.array], init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    if PROBLEM_NAME == ProblemName.maxcut:
        curr_score = obj_maxcut(curr_solution, graph)
    elif PROBLEM_NAME == ProblemName.graph_partitioning:
        curr_score = obj_graph_partitioning(curr_solution, graph)
    elif PROBLEM_NAME == ProblemName.minimum_vertex_cover:
        curr_score = obj_minimum_vertex_cover(curr_solution, graph)
        edges = list(graph.edges)
    init_score = curr_score
    num_nodes = len(init_solution)
    scores = []
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        new_solution = copy.deepcopy(curr_solution)
        if PROBLEM_NAME == ProblemName.maxcut:
            index = random.randint(0, num_nodes - 1)
            new_solution[index] = (new_solution[index] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
        elif PROBLEM_NAME == ProblemName.graph_partitioning:
            while True:
                index = random.randint(0, num_nodes - 1)
                index2 = random.randint(0, num_nodes - 1)
                if new_solution[index] != new_solution[index2]:
                    break
            print(f"new_solution[index]: {new_solution[index]}, new_solution[index2]: {new_solution[index2]}")
            tmp = new_solution[index]
            new_solution[index] = new_solution[index2]
            new_solution[index2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
        elif PROBLEM_NAME == ProblemName.minimum_vertex_cover:
            while True:
                index = random.randint(0, num_nodes - 1)
                if new_solution[index] == 0:
                    continue
                new_solution2 = copy.deepcopy(new_solution)
                new_solution2[index] = 0
                if cover_all_edges(new_solution2, graph):
                    break
            new_solution[index] = 0
            new_score = obj_minimum_vertex_cover(new_solution, graph)
        scores.append(new_score)
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
        else:
            prob = np.exp(- delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
    print("score, init_score of simulated_annealing", curr_score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # read data
    graph = read_nxgraph('../data/syn/syn_50_176.txt')

    # run alg
    # init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))
    if PROBLEM_NAME in [ProblemName.maxcut, ProblemName.graph_partitioning]:
        init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    if PROBLEM_NAME == ProblemName.minimum_vertex_cover:
        init_solution = [1] * int(graph.number_of_nodes())

    init_temperature = 4
    num_steps = 8000 * 3
    sa_score, sa_solution, sa_scores = simulated_annealing(init_solution, init_temperature, num_steps, graph)

    # write result
    write_result(sa_solution, '../result/result.txt')

    # plot fig
    alg_name = 'SA'
    plot_fig(sa_scores, alg_name)




import sys
sys.path.append('../')

import copy
import time
from typing import List, Union, Optional
import numpy as np
import multiprocessing as mp
import networkx as nx
from util import read_nxgraph
from util import (obj_maxcut,
                  obj_graph_partitioning,
                  obj_minimum_vertex_cover,
                  obj_maximum_independent_set,
                  obj_set_cover_ratio,
                  )
from util import plot_fig
from util import transfer_nxgraph_to_weightmatrix
from util import cover_all_edges
from util import run_greedy_over_multiple_files
from config import *

def split_list(my_list: List[int], chunk_size: int):
    res = []
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

# the len of result may not be mp.cpu_count()
def split_list_equally_by_cpus(my_list: List[int]):
    res = []
    num_cpus = mp.cpu_count()
    chunk_size = int(np.ceil(len(my_list) / num_cpus))
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

def split_list_equally(my_list: List[int], chunk_size: int):
    res = []
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

def traverse_in_greedy_maxcut(curr_solution, selected_nodes, graph):
    traversal_scores = []
    traversal_solutions = []
    # calc the new solution when moving to a new node. Then store the scores and solutions.
    for node in selected_nodes:
        new_solution = copy.deepcopy(curr_solution)
        # search a new solution and calc obj
        new_solution[node] = (new_solution[node] + 1) % 2
        new_score = obj_maxcut(new_solution, graph)
        traversal_scores.append(new_score)
        traversal_solutions.append(new_solution)
    return traversal_scores, traversal_solutions


# init_solution is useless
def greedy_maxcut(num_steps: Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    init_solution = [0] * graph.number_of_nodes()
    assert sum(init_solution) == 0
    if num_steps is None:
        num_steps = num_nodes
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        score = obj_maxcut(curr_solution, graph)
        print(f"iteration: {iteration}, score: {score}")
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        use_multiprocessing = False
        if use_multiprocessing:
            pass
            split_nodess = split_list_equally_by_cpus(nodes)
            pool = mp.Pool(len(split_nodess))
            # print(f'len split_nodess: {len(split_nodess)}')
            results = []
            for split_nodes in split_nodess:
                results.append(pool.apply_async(traverse_in_greedy_maxcut, (curr_solution, split_nodes, graph)))
            for result in results:
                tmp_traversal_scores, tmp_traversal_solutions = result.get()
                # print(f'tmp_traversal_scores: {tmp_traversal_scores}, tmp_traversal_solutions: {tmp_traversal_solutions}')
                traversal_scores.extend(tmp_traversal_scores)
                traversal_solutions.extend(tmp_traversal_solutions)
        else:
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
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_graph_partitioning(num_steps:Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    num_nodes = int(graph.number_of_nodes())
    if num_steps is None:
        num_steps = num_nodes
    start_time = time.time()
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_graph_partitioning(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_steps):
        node1 = nodes[i]
        traversal_scores = []
        traversal_solutions = []
        for j in range(i + 1, num_steps):
            node2 = nodes[j]
            if curr_solution[node1] == curr_solution[node2]:
                continue
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
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


def greedy_minimum_vertex_cover(num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    init_solution = [0] * graph.number_of_nodes()
    assert sum(init_solution) == 0
    assert num_steps is None
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    scores = []
    iter = 0
    unselected_nodes = list(graph.nodes())
    while True:
        cover_all = cover_all_edges(curr_solution, graph)
        if cover_all:
            break
        max_degree = 0
        best_node = -INF
        for node in unselected_nodes:
            degree = graph.degree(node)
            if degree > max_degree:
                max_degree = degree
                best_node = node
        if max_degree > 0:
            curr_solution[best_node] = 1
            unselected_nodes.remove(best_node)
        iter += 1
        if iter > num_nodes:
            break
    curr_score = obj_minimum_vertex_cover(curr_solution, graph)
    print("score, init_score", curr_score, init_score)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_maximum_independent_set(num_steps: Optional[int], graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    def calc_candidate_nodes(unselected_nodes: List[int], selected_nodes: List[int], graph: nx.Graph):
        candidate_nodes = []
        remove_nodes = set()
        for node1, node2 in graph.edges():
            if node1 in selected_nodes:
                remove_nodes.add(node2)
            elif node2 in selected_nodes:
                remove_nodes.add(node1)
        for node in unselected_nodes:
            if node not in remove_nodes:
                candidate_nodes.append(node)
        return candidate_nodes
    print('greedy')
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    init_solution = [0] * num_nodes
    if num_steps is None:
        num_steps = num_nodes
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maximum_independent_set(curr_solution, graph)
    init_score = curr_score
    scores = []
    selected_nodes = []
    unselected_nodes = copy.deepcopy(nodes)
    candidate_graph = copy.deepcopy(graph)
    # extend_candidate_graph = copy.deepcopy(graph)
    step = 0
    while True:
        step += 1
        candidate_nodes = calc_candidate_nodes(unselected_nodes, selected_nodes, graph)
        if len(candidate_nodes) == 0:
            break
        min_degree = num_nodes
        selected_node = None
        for node in candidate_nodes:
            degree = candidate_graph.degree(node)
            if degree < min_degree:
                min_degree = degree
                selected_node = node
        if selected_node is None:
            break
        else:
            selected_nodes.append(selected_node)
            unselected_nodes.remove(selected_node)
            candidate_graph.remove_node(selected_node)
            curr_solution[selected_node] = 1
            # curr_score2 = obj_maximum_independent_set(curr_solution, graph)
            curr_score += 1
            # assert curr_score == curr_score2
            scores.append(curr_score)
        if step > num_steps:
            break
    curr_score2 = obj_maximum_independent_set(curr_solution, graph)
    assert curr_score == curr_score2
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_set_cover(num_sets: int, num_items: int, item_matrix: List[List[int]]) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    curr_solution = [0] * num_sets
    init_score = 0.0
    curr_score = 0.0
    scores = []
    selected_sets = []
    unselected_sets = set(np.array(range(num_sets)) + 1)
    unselected_items = set(np.array(range(num_items)) + 1)
    while len(unselected_items) > 0:
        max_intersection_num = 0
        selected_set = None
        for i in unselected_sets:
            intersection_num = 0
            for j in item_matrix[i]:
                if j in unselected_items:
                    intersection_num += 1
            if intersection_num > max_intersection_num:
                max_intersection_num = intersection_num
                selected_set = i
        if selected_set is not None:
            selected_sets.append(selected_set)
            unselected_sets.remove(selected_set)
            for j in item_matrix[selected_set]:
                if j in unselected_items:
                    unselected_items.remove(j)
            curr_score += max_intersection_num / num_items
            scores.append(curr_score)
            curr_solution[selected_set - 1] = 1
    print("init_score, final score of greedy", init_score, curr_score, )
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # read data
    print(f'problem: {PROBLEM}')
    graph = read_nxgraph('../data/syn/syn_10_21.txt')
    weightmatrix = transfer_nxgraph_to_weightmatrix(graph)
    # run alg
    num_steps = 30
    alg_name = 'GR'

    if_run_one_case = False
    if if_run_one_case:
        # maxcut
        if PROBLEM == Problem.maxcut:
            gr_score, gr_solution, gr_scores = greedy_maxcut(num_steps, graph)

        # graph_partitioning
        elif PROBLEM == Problem.graph_partitioning:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_graph_partitioning(num_steps, graph)

        elif PROBLEM == Problem.minimum_vertex_cover:
            gr_score, gr_solution, gr_scores = greedy_minimum_vertex_cover(num_steps, graph)
            obj = obj_minimum_vertex_cover(gr_solution, graph)
            print('obj: ', obj)

        elif PROBLEM == Problem.maximum_independent_set:
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_maximum_independent_set(num_steps, graph)
            obj = obj_maximum_independent_set(gr_solution, graph)
            print('obj: ', obj)

    else:
        if PROBLEM == Problem.maxcut:
            alg = greedy_maxcut
        elif PROBLEM == Problem.graph_partitioning:
            alg = greedy_graph_partitioning
        elif PROBLEM == Problem.minimum_vertex_cover:
            alg = greedy_minimum_vertex_cover
        elif PROBLEM == Problem.maximum_independent_set:
            alg = greedy_maximum_independent_set
        elif PROBLEM == Problem.set_cover:
            alg = greedy_set_cover

        alg_name = "greedy"
        num_steps = None
        # directory_data = '../data/syn_BA'
        directory_data = '../data/syn_ER'
        # directory_data = '../data/syn'
        # prefixes = ['barabasi_albert_100_']
        prefixes = ['erdos_renyi_100_']
        # prefixes = ['syn_10_']

        if_run_set_cover = True
        if if_run_set_cover:
            directory_data = 'data/set_cover/frb45-21-5.msc'
            prefixes = ['frb45-21-5.msc']
        scoress = run_greedy_over_multiple_files(alg, alg_name, num_steps, directory_data, prefixes)
        print(f"scoress: {scoress}")

        # plot fig
        plot_fig_ = False
        if plot_fig_:
            for scores in scoress:
                plot_fig(scores, alg_name)





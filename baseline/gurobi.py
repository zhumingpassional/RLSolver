from gurobipy import *
import os
from typing import List
import networkx as nx
import sys
from util import read_nxgraph
from util import calc_txt_files_with_prefix
from util import calc_result_file_name
from util import calc_avg_std_of_objs
from util import plot_fig
from util import fetch_node
from util import transfer_float_to_binary

# the file has been open
def write_statistics(model, new_file, add_slash = False):
    prefix = '// ' if add_slash else ''
    new_file.write(f"{prefix}obj: {model.objVal}\n")
    new_file.write(f"{prefix}running_duration: {model.Runtime}\n")
    new_file.write(f"{prefix}gap: {model.MIPGap}\n")
    new_file.write(f"{prefix}obj_bound: {model.ObjBound}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    new_file.write(f"{prefix}time_limit: {time_limit}\n")

# running_duration (seconds) is included.
def write_result_gurobi(model, filename: str = './result/result', running_duration: int = None):
    if 'data' in filename:
        filename = calc_result_file_name(filename)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    if running_duration is None:
        new_filename = filename
    else:
        filename = filename.replace('.txt', '')
        new_filename = filename + '_' + str(int(running_duration))

    vars = model.getVars()
    nodes: List[int] = []
    values: List[int] = []
    for var in vars:
        node = fetch_node(var.VarName)
        if node is None:
            break
        value = transfer_float_to_binary(var.x)
        nodes.append(node)
        values.append(value)
    with open(f"{new_filename}.txt", 'w', encoding="UTF-8") as new_file:
        write_statistics(model, new_file, True)
        new_file.write(f"num_nodes: {len(nodes)}\n")
        for i in range(len(nodes)):
            new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")

    if_write_others = False
    if if_write_others:
        with open(f"{new_filename}.sta", 'w', encoding="UTF-8") as new_file:
            write_statistics(model, new_file, False)
        with open(f"{new_filename}.sov", 'w', encoding="UTF-8") as new_file:
            new_file.write('values of vars: \n')
            vars = model.getVars()
            for var in vars:
                new_file.write(f'{var.VarName}: {var.x}\n')
        model.write(f"{new_filename}.mst")
        model.write(f"{new_filename}.lp")
        model.write(f"{new_filename}.mps")
        model.write(f"{new_filename}.sol")

def run_using_gurobi(filename: str, time_limit: int = None, plot_fig_: bool = False):
    model = Model("maxcut")

    graph = read_nxgraph(filename)

    adjacency_matrix = nx.to_numpy_array(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="y")
    model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                    GRB.MAXIMIZE)


    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addConstr(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addConstr(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))
    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('../result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        # result_filename = '../result/result'
        result_filename = calc_result_file_name(filename)
        write_result_gurobi(model, result_filename, time_limit)

    num_vars = model.getAttr(GRB.Attr.NumVars)
    num_constrs = model.getAttr(GRB.Attr.NumConstrs)
    print(f'num_vars: {num_vars}, num_constrs: {num_constrs}')
    print('obj:', model.getObjective().getValue())
    vars = model.getVars()


    if model.getAttr('SolCount') == 0:  # model.getAttr(GRB.Attr.SolCount)
        print("No solution.")
    print("SolCount: ", model.getAttr('SolCount'))
    # except Exception as e:
    #     print("Exception!")

    scores = [model.getObjective().getValue()]
    alg_name = 'Gurobi'
    if plot_fig_:
        plot_fig(scores, alg_name)
    print()

def run_gurobi_over_multiple_files(prefixes: List[str], time_limits: List[int], directory_data: str = 'data', directory_result: str = 'result'):
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        for i in range(len(files)):
            print(f'The {i}-th file: {files[i]}')
            for j in range(len(time_limits)):
                run_using_gurobi(files[i], time_limits[j])
    avg_std = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

if __name__ == '__main__':
    select_single_file = False
    if select_single_file:
        filename = '../data/syn/syn_50_176.txt'
        time_limits = [0.5 * 3600]
        run_using_gurobi(filename, time_limit=time_limits[0], plot_fig_=False)
        directory = '../result'
        prefixes = ['syn_50_']
        avg_std = calc_avg_std_of_objs(directory, prefixes, time_limits)
    else:
        if_use_syn = False
        # time_limits = [0.5 * 3600]
        time_limits = [10 * 60, 20 * 60, 30 * 60, 40 * 60, 50 * 60, 60 * 60]
        if if_use_syn:
            prefixes = ['syn_10_', 'syn_50_', 'syn_100_', 'syn_300_', 'syn_500_', 'syn_700_', 'syn_900_', 'syn_1000_', 'syn_3000_', 'syn_5000_', 'syn_7000_', 'syn_9000_', 'syn_10000_']
            directory_data = '../data/syn'

        if_use_syndistri = True
        if if_use_syndistri:
            prefixes = ['powerlaw_20_']
            directory_data = '../data/syndistri2'

        directory_result = '../result'
        run_gurobi_over_multiple_files(prefixes, time_limits, directory_data, directory_result)
        avg_std = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

    pass


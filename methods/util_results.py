import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
from torch.autograd import Variable
import os
import functools
import time
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple, Optional
import networkx as nx
import pandas as pd
import torch as th
from torch import Tensor
from os import system
import math
from enum import Enum
import tqdm
import re
# from methods.simulated_annealing import simulated_annealing_set_cover, simulated_annealing
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# return: num_nodes, ID, running_duration:, obj,
def read_result_comments(filename: str):
    num_nodes, ID, running_duration, obj = None, None, None, None
    ID = int(filename.split('ID')[1].split('_')[0])
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        while line is not None and line != '':
            if '//' in line:
                if 'num_nodes:' in line:
                    num_nodes = float(line.split('num_nodes:')[1])
                    break
                if 'running_duration:' in line:
                    running_duration = obtain_first_number(line)
                if 'obj:' in line:
                    obj = float(line.split('obj:')[1])
            line = file.readline()
    return int(num_nodes), ID, running_duration, obj

def read_result_comments_multifiles2(dir: str, prefixes: str, max_ID: int):
    objs = {}
    running_durations = {}
    # for prefix in prefixes:
    files = calc_txt_files_with_prefix(dir, prefixes)
    for i in range(len(files)):
        file = files[i]
        num_nodes, ID, running_duration, obj = read_result_comments(file)
        if ID >= max_ID + 1:
            continue
        if str(num_nodes) not in objs.keys():
            objs[str(num_nodes)] = [obj]
            running_durations[str(num_nodes)] = [running_duration]
        else:
            objs[str(num_nodes)].append(obj)
            running_durations[str(num_nodes)].append(running_duration)

    label = f"num_nodes={num_nodes}"
    print(f"objs: {objs}, running_durations: {running_durations}")
    # objs = [(key, objs[key]) for key in sorted(objs.keys())]
    objs = dict(sorted(objs.items(), key=lambda x: x[0]))
    running_durations = dict(sorted(running_durations.items(), key=lambda x: x[0]))
    avg_objs = {}
    avg_running_durations = {}
    std_objs = {}
    std_running_durations = {}
    for key, value in objs.items():
        avg_objs[key] = np.average(value)
        std_objs[key] = np.std(value)
    for key, value in running_durations.items():
        avg_running_durations[key] = np.average(value)
        std_running_durations[key] = np.std(value)
    return objs, running_durations, avg_objs, avg_running_durations, std_objs, std_running_durations



def calc_txt_files_with_prefix(directory: str, prefix: str):
    res = []
    files = os.listdir(directory)
    for file in files:
        if prefix in file and ('.txt' in file or '.msc' in file):
            res.append(directory + '/' + file)
    return res


# e.g., s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)",
# then returns 36
def obtain_first_number(s: str):
    res = ''
    pass_first_digit = False
    for i in range(len(s)):
        if s[i].isdigit() or s[i] == '.':
            res += s[i]
            pass_first_digit = True
        elif pass_first_digit:
            break
    value = int(float(res))
    return value







if __name__ == '__main__':
    # dir = 'syn_BA_greedy2approx'
    dir = '../result/syn_ER_gurobi'
    prefixes = 'erdos_renyi_'
    max_ID = 9
    objs, running_durations, avg_objs, avg_running_durations, std_objs, std_running_durations = read_result_comments_multifiles2(dir, prefixes, max_ID)




    print(avg_objs)
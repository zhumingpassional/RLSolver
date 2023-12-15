import torch as th
from typing import List, Union, Tuple
from enum import Enum, unique

@unique
class Problem(Enum):
    maxcut = 'maxcut'
    graph_partitioning = 'graph_partitioning'
    minimum_vertex_cover = 'minimum_vertex_cover'

@unique
class GraphDistriType(Enum):
    erdos_renyi: str = 'erdos_renyi'
    powerlaw: str = 'powerlaw'
    barabasi_albert: str = 'barabasi_albert'

def calc_device(gpu_id: int):
    return th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

GPU_ID: int = 0  # -1: cpu, >=0: gpu
DEVICE: th.device = calc_device(GPU_ID)
DATA_DIR: str = './data'
GSET_DIR: str = './data/gset'
GRAPH_DISTRI_TYPE = GraphDistriType.powerlaw
GRAPH_DISTRI_TYPES: List[GraphDistriType] = [GraphDistriType.erdos_renyi, GraphDistriType.powerlaw, GraphDistriType.barabasi_albert]
    # graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']
NUM_IDS = 30  # ID0, ..., ID29

PROBLEM = Problem.maxcut

INF = 1e6

# RUNNING_DURATIONS = [600, 1200, 1800, 2400, 3000, 3600]  # store results
RUNNING_DURATIONS = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]  # store results

GUROBI_INTERVAL = 10 * 60  # seconds, the interval of writing results to txt files
GUROBI_TIME_LIMITS = [1 * 3600]  # seconds
# GUROBI_TIME_LIMITS = [600, 1200, 1800, 2400, 3000, 3600]  # seconds
# GUROBI_TIME_LIMITS2 = list(range(10 * 60, 1 * 3600 + 1, 10 * 60))  # seconds
GUROBI_VAR_CONTINUOUS = False




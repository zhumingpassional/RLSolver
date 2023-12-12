import torch as th
from typing import List, Union, Tuple
from enum import Enum, unique

@unique
class ProblemName(Enum):
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

PROBLEM_NAME = ProblemName.graph_partitioning

INF = 1e9

GUROBI_INTERVAL = 10 * 60  # seconds, the interval of writing results to txt files
GUROBI_TIME_LIMITS = [1 * 3600]  # seconds
GUROBI_VAR_CONTINUOUS = False




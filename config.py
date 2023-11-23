import torch as th
from util import calc_device
from typing import List, Union, Tuple
class GraphDistriType:
    erdos_renyi: str = 'erdos_renyi'
    powerlaw: str = 'powerlaw'
    barabasi_albert: str = 'barabasi_albert'


class Config:
    gpu_id: int = 0  # -1: cpu, >=0: gpu
    device: th.device = calc_device(gpu_id)
    data_dir: str = './data'
    gset_dir: str = './data/gset'
    graph_distri_types: List[GraphDistriType] = [GraphDistriType.erdos_renyi, GraphDistriType.powerlaw, GraphDistriType.barabasi_albert]
    # graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']



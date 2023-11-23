from util import GraphDistriType
class Config:
    gpu_id = 0
    data_dir = './data'
    gset_dir = './data/gset'
    graph_distri_types = [GraphDistriType.erdos_renyi, GraphDistriType.powerlaw, GraphDistriType.barabasi_albert]
    # graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']


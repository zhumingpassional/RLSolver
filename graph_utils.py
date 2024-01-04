import os
import sys
import torch as th
import networkx as nx
from typing import List, Tuple

'''graph'''

TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]
DataDir = './data/graph_max_cut'

'''load graph'''


def load_graph_list_from_txt(txt_path: str = 'G14.txt') -> GraphList:
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph_list = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # 将node_id 由“从1开始”改为“从0开始”

    assert num_nodes == obtain_num_nodes(graph_list=graph_list)
    assert num_edges == len(graph_list)
    return graph_list


def generate_graph_list(graph_type: str, num_nodes: int) -> GraphList:
    graph_types = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert']
    assert graph_type in graph_types

    if graph_type == 'ErdosRenyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == 'PowerLaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif graph_type == 'BarabasiAlbert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    distance = 1
    graph_list = [(node0, node1, distance) for node0, node1 in g.edges]
    return graph_list


def load_graph_list(graph_name: str):
    import random
    graph_types = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert']
    graph_type = next((graph_type for graph_type in graph_types if graph_type in graph_name), None)

    if os.path.exists(f"{DataDir}/{graph_name}.txt"):
        txt_path = f"{DataDir}/{graph_name}.txt"
        graph_list = load_graph_list_from_txt(txt_path=txt_path)
    elif graph_type and graph_name.find('ID') == -1:
        num_nodes = int(graph_name.split('_')[-1])
        graph_list = generate_graph_list(num_nodes=num_nodes, graph_type=graph_type)
    elif graph_type and graph_name.find('ID') >= 0:
        num_nodes, valid_i = graph_name.split('_')[-2:]
        num_nodes = int(num_nodes)
        valid_i = int(valid_i[len('ID'):])
        random.seed(valid_i)
        graph_list = generate_graph_list(num_nodes=num_nodes, graph_type=graph_type)
        random.seed()
    elif os.path.isfile(graph_name):
        txt_path = graph_name
        graph_list = load_graph_list_from_txt(txt_path=txt_path)
    else:
        raise ValueError(f"DataDir {DataDir} | graph_name {graph_name}")
    return graph_list




'''adjacency matrix'''


def build_adjacency_matrix(graph_list: GraphList, if_bidirectional: bool = False):
    """例如，无向图里：
    - 节点0连接了节点1
    - 节点0连接了节点2
    - 节点2连接了节点3

    用邻接阶矩阵Ary的上三角表示这个无向图：
      0 1 2 3
    0 F T T F
    1 _ F F F
    2 _ _ F T
    3 _ _ _ F

    其中：
    - Ary[0,1]=True
    - Ary[0,2]=True
    - Ary[2,3]=True
    - 其余为False
    """
    not_connection = -1  # 选用-1去表示表示两个node之间没有edge相连，不选用0是为了避免两个节点的距离为0时出现冲突
    num_nodes = obtain_num_nodes(graph_list=graph_list)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph_list:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix


def build_adjacency_indies(graph_list: GraphList, if_bidirectional: bool = False) -> (IndexList, IndexList):
    """
    用二维列表list2d表示这个图：
    [
        [1, 2],
        [],
        [3],
        [],
    ]
    其中：
    - list2d[0] = [1, 2]
    - list2d[2] = [3]

    对于稀疏的矩阵，可以直接记录每条边两端节点的序号，用shape=(2,N)的二维列表 表示这个图：
    0, 1
    0, 2
    2, 3
    如果条边的长度为1，那么表示为shape=(2,N)的二维列表，并在第一行，写上 4个节点，3条边的信息，帮助重建这个图，然后保存在txt里：
    4, 3
    0, 1, 1
    0, 2, 1
    2, 3, 1
    """
    num_nodes = obtain_num_nodes(graph_list=graph_list)

    n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
    n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
    for n0, n1, distance in graph_list:
        n0_to_n1s[n0].append(n1)
        n0_to_dts[n0].append(distance)
        if if_bidirectional:
            n0_to_n1s[n1].append(n0)
            n0_to_dts[n1].append(distance)
    n0_to_n1s = [th.tensor(node1s) for node1s in n0_to_n1s]
    n0_to_dts = [th.tensor(node1s) for node1s in n0_to_dts]
    assert num_nodes == len(n0_to_n1s)
    assert num_nodes == len(n0_to_dts)

    '''sort'''
    for i, node1s in enumerate(n0_to_n1s):
        sort_ids = th.argsort(node1s)
        n0_to_n1s[i] = n0_to_n1s[i][sort_ids]
        n0_to_dts[i] = n0_to_dts[i][sort_ids]
    return n0_to_n1s, n0_to_dts


'''get_hot_tensor_of_graph'''


def show_array2d(show_array, title='array2d', if_save=False):
    import matplotlib.pyplot as plt

    plt.cla()
    plt.imshow(show_array, cmap='hot', interpolation='nearest')
    plt.colorbar(label='hot map')
    plt.title(title)
    plt.tight_layout()

    if if_save:
        plt.savefig(f"hot_image_{title}.jpg", dpi=400)
        plt.close('all')
    else:
        plt.show()


def get_hot_tensor_of_graph(adj_matrix):
    from math import log as math_log
    num_nodes = adj_matrix.size(0)
    num_iters = int(math_log(num_nodes))
    device = adj_matrix.device

    hot_matrix = adj_matrix.clone()
    adjust_eye = th.eye(num_nodes, device=device)
    for i in range(num_iters):
        hot_matrix = th.matmul(hot_matrix + adjust_eye, adj_matrix)
        hot_matrix = hot_matrix + hot_matrix.t()
    return th.log(hot_matrix.clip(1e-12, 1e+12)) / math_log(num_nodes)


def check_get_hot_tenor_of_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    if_save = True

    graph_names = []
    for graph_type in ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert']:
        for num_nodes in (128, 1024):
            for seed_id in range(2):
                graph_names.append(f'{graph_type}_{num_nodes}_ID{seed_id}')
    for gset_id in (14, 15, 49, 50, 22, 55, 70):
        graph_names.append(f"gset_{gset_id}")

    for graph_name in graph_names:
        graph_list: GraphList = load_graph_list(graph_name=graph_name)

        graph = nx.Graph()
        for n0, n1, distance in graph_list:
            graph.add_edge(n0, n1, weight=distance)

        adj_matrix = nx.adjacency_matrix(graph).toarray()
        hot_array = get_hot_tensor_of_graph(adj_matrix=th.tensor(adj_matrix, device=device)).cpu().data.numpy()

        title = f"{graph_name}_N{graph.number_of_nodes()}_E{graph.number_of_edges()}"
        print(f"title {title}")
        show_array2d(show_array=hot_array, title=title, if_save=if_save)
    print()


'''utils'''


def obtain_num_nodes(graph_list: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph_list]) + 1


def get_gpu_info_str(device) -> str:
    if not th.cuda.is_available():
        return 'th.cuda.is_available() == False'

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    return (f"RAM(GB) {memory_allocated:.2f} < {max_allocated:.2f} < {total_memory:.2f}  "
            f"Rate {(max_allocated / total_memory):5.2f}")


if __name__ == '__main__':
    check_get_hot_tenor_of_graph()

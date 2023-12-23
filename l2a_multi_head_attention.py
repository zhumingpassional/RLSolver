import os
import sys
import time
import tqdm
import torch as th
import torch.nn as nn
import networkx as nx
from typing import List, Tuple
from torch.nn.utils import clip_grad_norm_

'''graph'''
TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]


def generate_graph(graph_type: str, num_nodes: int) -> GraphList:
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']
    assert graph_type in graph_types

    if graph_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif graph_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    distance = 1
    graph = [(node0, node1, distance) for node0, node1 in g.edges]
    return graph


def obtain_num_nodes(graph: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph]) + 1


def build_adjacency_matrix(graph: GraphList, if_bidirectional: bool = False):
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
    num_nodes = obtain_num_nodes(graph=graph)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix


'''net'''


class BnMLP(nn.Module):
    def __init__(self, dims, activation=None):
        super(BnMLP, self).__init__()

        assert len(dims) >= 3
        mlp_list = [nn.Linear(dims[0], dims[1]), ]
        for i in range(1, len(dims) - 1):
            dim_i = dims[i]
            dim_j = dims[i + 1]
            mlp_list.extend([nn.GELU(), nn.LayerNorm(dim_i), nn.Linear(dim_i, dim_j)])

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, inp):
        return self.mlp(inp)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class AttentionNet(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads):
        super(AttentionNet, self).__init__()
        self.encoder_net = BnMLP(dims=(inp_dim, mid_dim, mid_dim, embed_dim), activation=nn.GELU())
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.decoder_net = BnMLP(dims=(embed_dim, mid_dim, mid_dim, out_dim), activation=nn.Tanh())

    def forward(self, inp_seq, mask=None):
        enc_seq = self.encoder_net(inp_seq)
        mid_seq, attention_weight = self.attention(enc_seq, enc_seq, enc_seq, attn_mask=mask)
        dec_seq = self.decoder_net(mid_seq)
        return dec_seq

    def encode_adjacency_matrix(self, inp_seq, mask=None):
        enc_seq = self.encoder_net(inp_seq)
        mid_seq, attention_weight = self.attention(enc_seq, enc_seq, enc_seq, attn_mask=mask)
        return mid_seq


def create_mask(seq_len, mask_type):
    if mask_type == 'triu':
        # Create an upper triangular matrix with ones above the diagonal
        mask = th.triu(th.ones(seq_len, seq_len), diagonal=1)
    elif mask_type == 'eye':
        # Create a square matrix with zeros on the diagonal
        mask = th.eye(seq_len)
    else:
        raise ValueError("type should in ['triu', 'eye']")
    return mask.masked_fill(mask == 1, float('-inf'))


def check_net():
    # Example usage:
    num_nodes = 100
    num_heads = 8
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % 8
    batch_size = 3

    net = AttentionNet(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, embed_dim=embed_dim, num_heads=num_heads)

    seq_len = num_nodes
    mask = create_mask(seq_len, mask_type='eye')
    input_tensor = th.rand(seq_len, batch_size, inp_dim)
    output_tensor = net(input_tensor, mask)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)


'''train'''


def generate_adjacency_seq(num_sims, graph_type, num_nodes, if_tqdm=False):
    adjacency_seq = th.empty((num_nodes, num_sims, num_nodes), dtype=th.float32)

    i_iteration = tqdm.trange(num_sims, ascii=True) if if_tqdm else range(num_sims)
    for i in i_iteration:
        graph = generate_graph(graph_type=graph_type, num_nodes=num_nodes)
        adjacency_ary = build_adjacency_matrix(graph=graph, if_bidirectional=True)
        adjacency_seq[:, i, :] = adjacency_ary
    return adjacency_seq


def train_attention_net_in_graph_dist():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = 'powerlaw'
    num_sims = 64
    buf_size = num_sims * 32
    update_size = 8
    net_path = f"./attention_net_{graph_type}_Node{num_nodes}_{gpu_id}.pth"
    buf_path = f"./buffer_{graph_type}_Node{num_nodes}_Buffer{buf_size}.pth"
    show_gap = 4

    if os.name == 'nt':
        num_sims = 8
        buf_size = num_sims * 4
        update_size = 2

    '''model'''
    num_heads = 16
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % 8
    learning_rate = 1e-3
    weight_decay = 1e-5

    net = AttentionNet(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                       embed_dim=embed_dim, num_heads=num_heads).to(device)
    mask = create_mask(num_nodes, mask_type='eye').to(device)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''buffer'''
    print(f"| generate_adjacency_seq")
    adjacency_buf = th.empty((num_nodes, buf_size, num_nodes), dtype=th.float32, device=device)
    if os.path.exists(buf_path):
        buf = th.load(buf_path, map_location=device)
        load_size = min(buf.shape, buf_size)
        adjacency_buf[:, :load_size, :] = buf[:, :load_size, :]
    else:
        load_size = 0
    generate_size = buf_size - load_size
    if generate_size > 0:
        adjacency_seq = generate_adjacency_seq(num_sims=generate_size, if_tqdm=True,
                                               graph_type=graph_type, num_nodes=num_nodes).to(device)
        adjacency_buf[:, load_size:, :] = adjacency_seq
    if buf_size > load_size:
        th.save(adjacency_buf, buf_path)

    timer = time.time()
    for j in range(128):
        rand_is = th.randint(0, adjacency_buf.shape[1], size=(num_sims,), device=device)

        '''update adjacency_buf'''
        adjacency_seq = generate_adjacency_seq(num_sims=update_size,
                                               graph_type=graph_type, num_nodes=num_nodes).to(device)
        adjacency_buf[:, rand_is[:update_size], :] = adjacency_seq

        '''update network'''
        inp = adjacency_buf[:, rand_is, :]
        out = net(inp, mask)
        objective = criterion(out, inp.detach())

        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        if j % show_gap == 0:
            exec_time = int(time.time() - timer)
            inp = adjacency_buf[:, rand_is[:update_size], :]
            out = net(inp, mask)
            objective = criterion(out, inp.detach())
            print(f"| {j:4}  {exec_time:4} sec | obj {objective.item():9.3f}")
    th.save(net.state_dict(), net_path)
    print(f"| save net in {net_path}")


if __name__ == '__main__':
    train_attention_net_in_graph_dist()

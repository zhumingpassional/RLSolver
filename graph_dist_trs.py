import os
import sys
import time
import tqdm
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from graph_max_cut_mh_sampling import GraphTRS, PolicyTRS, create_mask
from graph_utils import generate_graph_list, build_adjacency_matrix
from graph_utils import get_hot_tensor_of_graph


def generate_adjacency_seq(num_sims, graph_type, num_nodes, if_tqdm=False):
    adjacency_seq = th.empty((num_nodes, num_sims, num_nodes), dtype=th.bool)

    i_iteration = tqdm.trange(num_sims, ascii=True) if if_tqdm else range(num_sims)
    for i in i_iteration:
        graph = generate_graph_list(graph_type=graph_type, num_nodes=num_nodes)
        adjacency_ary = build_adjacency_matrix(graph_list=graph, if_bidirectional=True)
        adjacency_seq[:, i, :] = adjacency_ary.eq(1)
    return adjacency_seq


def get_buffer(train_inp_path, buf_size, graph_type, num_nodes, device):
    train_inp = th.empty((num_nodes, buf_size, num_nodes), dtype=th.bool, device=device)
    if os.path.exists(train_inp_path):
        inp = th.load(train_inp_path, map_location=device)
        load_size = min(inp.shape[1], buf_size)
        train_inp[:, :load_size, :] = inp[:, :load_size, :]
    else:
        load_size = 0
    generate_size = buf_size - load_size
    if generate_size > 0:
        inp = generate_adjacency_seq(num_sims=generate_size, if_tqdm=True,
                                     graph_type=graph_type, num_nodes=num_nodes).to(device)
        train_inp[:, load_size:, :] = inp
    if buf_size > load_size:
        th.save(train_inp, train_inp_path)

    rand_is = th.randperm(buf_size, device=device)
    return train_inp[:, rand_is, :]


def train_graph_trs_net():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config'''
    # num_nodes = 500
    # graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][1]
    # num_buffer = 4
    num_nodes, graph_type, num_buffer = [
        [100, 'ErdosRenyi', 2], [100, 'PowerLaw', 4], [100, 'BarabasiAlbert', 3],
        [200, 'ErdosRenyi', 2], [200, 'PowerLaw', 4], [200, 'BarabasiAlbert', 3],
        [300, 'ErdosRenyi', 2], [300, 'PowerLaw', 4], [300, 'BarabasiAlbert', 3],
        [400, 'ErdosRenyi', 2], [400, 'PowerLaw', 4], [400, 'BarabasiAlbert', 3],
        [500, 'ErdosRenyi', 2], [500, 'PowerLaw', 4], [500, 'BarabasiAlbert', 3],
        [600, 'ErdosRenyi', 2], [600, 'PowerLaw', 4], [600, 'BarabasiAlbert', 3],
    ][gpu_id]

    num_sims = 32
    buf_size = 2 ** 12
    num_epochs = buf_size // num_sims
    net_path = f"./graph_net_{graph_type}_Node{num_nodes}.pth"
    show_gap = 32

    if os.name == 'nt':
        num_sims = 8
        buf_size = num_sims * 4

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads
    learning_rate = 2 ** -12
    weight_decay = 0

    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    th.save(net.state_dict(), net_path)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''buffer'''
    timer = time.time()
    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)

    def get_objective(_adj_matrix, if_train):
        if if_train:
            net.train()
            _mask = mask
        else:
            net.eval()
            _mask = None

        inp = _adj_matrix.float()
        lab1 = th.empty_like(inp)
        lab2 = th.empty_like(inp)
        for i in range(lab1.shape[1]):
            adj = _adj_matrix[:, i, :]
            lab1[:, i, :] = get_hot_tensor_of_graph(adj_matrix=adj / int(num_nodes ** 0.5))
            lab2[:, i, :] = get_hot_tensor_of_graph(adj_matrix=adj / (adj.sum(dim=1, keepdim=True) * 0.25))

        out = net(inp.float(), mask=_mask)
        out1, out2 = th.chunk(out, chunks=2, dim=2)
        _objective = criterion(out1, lab1.detach()) + criterion(out2, lab2.detach())
        return _objective

    for buffer_id in list(range(num_buffer + 1)) + list(range(num_buffer)) + list(range(num_buffer + 1)):
        with th.no_grad():
            dir_path = f"./buffer_{graph_type}_Node{num_nodes}"
            os.makedirs(dir_path, exist_ok=True)
            train_inp_path = f"./{dir_path}/buffer_{buffer_id:02}.pth"
            train_inp = get_buffer(train_inp_path, buf_size, graph_type, num_nodes, device)

        '''train loop'''
        for j in range(num_epochs):
            j0 = j * num_sims
            j1 = j0 + num_sims
            adj_matrix = train_inp[:, j0:j1, :]

            '''valid'''
            if j % show_gap == 0:
                objective = get_objective(adj_matrix, if_train=False)

                exec_time = int(time.time() - timer)
                print(f"| {buffer_id:2}  {j:4}  {exec_time:4} sec | obj {objective.item():9.4f}")

            '''train'''
            objective = get_objective(adj_matrix, if_train=True)

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()

    th.save(net.state_dict(), net_path)
    print(f"| save net in {net_path}")


def valid_graph_trs_net():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][1]
    num_sims = 16
    net_path = f"./attention_net_{graph_type}_Node{num_nodes}.pth"

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(net_path, map_location=device))

    inp = generate_adjacency_seq(num_sims=num_sims, graph_type=graph_type, num_nodes=num_nodes).to(device)
    dec_seq, dec_matrix, dec_node = graph_net.get_attn_matrix(inp.float(), mask=None)

    # from graph_utils import show_array2d
    # show_array2d(dec_node[:, 0, :].cpu().data.numpy())
    # show_array2d(dec_matrix[0, :, :].cpu().data.numpy())
    # print()

    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    prob = th.zeros(num_nodes, dtype=th.float32, device=device)
    prob = policy_net.auto_regression(prob, dec_node, dec_matrix)
    print(prob)


if __name__ == '__main__':
    train_graph_trs_net()
    # valid_graph_trs_net()

import os
import sys

import torch as th
import torch.nn as nn
import tqdm
from torch.nn.utils import clip_grad_norm_

from evaluator import Evaluator
from graph_max_cut_simulator import SimulatorGraphMaxCut
from graph_max_cut_local_search import SolverLocalSearch, show_gpu_memory
from graph_utils import load_graph_list_from_txt


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


class PolicyMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.net1 = BnMLP(dims=(inp_dim, mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

    def forward(self, xs):
        return self.net1(xs)

    def auto_regression(self, xs, ids_ary):
        num_sims, num_iter = ids_ary.shape
        sim_ids = th.arange(num_sims, device=xs.device)
        xs = xs.detach().clone()

        for i in range(num_iter):
            ids = ids_ary[:, i]

            ps = self.forward(xs.clone())[sim_ids, ids]
            xs[sim_ids, ids] = ps
        return xs


class PolicyRNN0(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_layers=num_layers)
        self.mlp_inp = BnMLP(dims=(inp_dim, mid_dim, mid_dim), activation=nn.GELU())
        self.rnn = nn.GRU(mid_dim, mid_dim, num_layers=num_layers)
        self.mlp_out = BnMLP(dims=(mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

    def forward(self, inp, hid=None):
        inp = self.mlp_inp(inp)
        rnn, hid = self.rnn(inp, hid)
        out = self.mlp_out(rnn)
        return out, hid


class PolicyRNN(nn.Module):  # todo
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(embed_dim + 1, mid_dim, embed_dim), activation=nn.GELU())
        self.rnn = nn.GRU(embed_dim, mid_dim, num_layers=num_layers)
        self.mlp_out = BnMLP(dims=(mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

    def forward(self, inp, hid=None):
        inp = self.mlp_inp(inp)
        rnn, hid = self.rnn(inp, hid)
        out = self.mlp_out(rnn)
        return out, hid

    def auto_regression(self, prob, dec_node, dec_matrix):
        # assert dec_node.shape == (num_nodes, batch_size, embed_dim)
        # assert dec_matrix.shape == (batch_size, mid_dim)
        i = 0

        inp = th.concat((dec_node, prob[:, None, None]), dim=2)
        out, hid = self.forward(inp=inp, hid=None)
        prob = out[:, i, 0]
        return prob


class GraphTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(inp_dim, inp_dim, mid_dim, embed_dim), activation=nn.GELU())

        self.trs_encoders = []
        for layer_id in range(num_layers):
            trs_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoders.append(trs_encoder)
            setattr(self, f'trs_encoder{layer_id:02}', trs_encoder)
        self.mlp_enc = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.trs_decoders = []
        for layer_id in range(num_layers):
            trs_decoder = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_decoders.append(trs_decoder)
            setattr(self, f'trs_decoder{layer_id:02}', trs_decoder)
        self.mlp_dec = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.num_nodes = inp_dim
        self.mlp_node = BnMLP(dims=(embed_dim, mid_dim, mid_dim, embed_dim), activation=nn.GELU())
        self.mlp_matrix = BnMLP(dims=(embed_dim, mid_dim, mid_dim, mid_dim), activation=None)
        self.activation = nn.GELU()

        self.mlp0 = BnMLP(dims=(mid_dim + embed_dim, mid_dim, out_dim, out_dim), activation=None)

    def forward(self, inp0, mask):
        enc1, dec_matrix, dec_node = self.get_attn_matrix(inp0, mask)

        dec1 = [trs_decoder(enc1, dec_node, mask, mask) for trs_decoder in self.trs_decoders]
        dec1 = self.mlp_dec(th.concat(dec1, dim=2))

        dec = th.concat((dec_matrix.repeat(self.num_nodes, 1, 1), dec1), dim=2)
        out = self.mlp0(dec)
        return out

    def get_attn_matrix(self, inp0, mask):
        enc0 = self.mlp_inp(inp0)

        enc1 = [trs_encoder(enc0, mask) for trs_encoder in self.trs_encoders]
        enc1 = self.mlp_enc(th.concat(enc1, dim=2))

        dec_matrix = self.mlp_matrix(enc1).sum(dim=0, keepdims=True)
        dec_matrix = self.activation(dec_matrix)
        dec_node = self.mlp_node(enc1)
        return enc1, dec_matrix, dec_node


class PolicyTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(embed_dim + 1, mid_dim, embed_dim), activation=nn.GELU())

        self.trs_encoders = []
        for layer_id in range(num_layers):
            trs_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoders.append(trs_encoder)
            setattr(self, f'trs_encoder{layer_id:02}', trs_encoder)
        self.mlp_enc = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.mlp_matrix = BnMLP(dims=(embed_dim, mid_dim, mid_dim, mid_dim), activation=None)
        self.activation = nn.GELU()

        # self.mlp0 = BnMLP(dims=(embed_dim + mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())
        self.mlp0 = BnMLP(dims=(mid_dim + embed_dim + mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

        self.num_nodes = inp_dim

    def forward(self, prob, dec_node):
        # assert prob.shape == (num_nodes, )
        # assert dec_node.shape == (num_nodes, embed_dim)

        inp0 = th.concat((dec_node, prob[:, None]), dim=1)[:, None, :]
        # assert inp0.shape == (num_nodes, 1, embed_dim+1)
        enc0 = self.mlp_inp(inp0)
        # assert enc0.shape == (num_nodes, 1, embed_dim)

        enc1 = [trs_encoder(enc0, src_mask=None) for trs_encoder in self.trs_encoders]
        enc1 = self.mlp_enc(th.concat(enc1, dim=2))

        dec_prob = self.mlp_matrix(enc1).sum(dim=0)
        dec_prob = self.activation(dec_prob)
        # assert dec_prob.shape == (mid_dim, )
        return dec_prob

    def auto_regression(self, prob, dec_node, dec_matrix):
        # assert dec_node.shape == (num_nodes, batch_size, embed_dim)
        # assert dec_matrix.shape == (batch_size, mid_dim)
        i = 0
        dec_matrix = dec_matrix[:, i, :]
        dec_node = dec_node[:, i, :]

        dec_prob = self.forward(prob.clone().detach(), dec_node.detach())
        dec = th.concat((dec_node, dec_matrix.repeat(self.num_nodes, 1), dec_prob.repeat(self.num_nodes, 1)), dim=1)
        prob = self.mlp0(dec).squeeze(1)
        return prob


def create_mask(seq_len, mask_type):
    if mask_type == 'triu':
        # Create an upper triangular matrix with ones above the diagonal
        mask = th.triu(th.ones(seq_len, seq_len), diagonal=1)
    elif mask_type == 'eye':
        # Create a square matrix with zeros on the diagonal
        mask = th.eye(seq_len)
    else:
        raise ValueError("type should in ['triu', 'eye']")
    return mask.bool()  # True means not allowed to attend.


def check_graph_trs_net():
    # Example usage:
    num_nodes = 100
    num_heads = 4
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads
    batch_size = 3

    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

    seq_len = num_nodes
    mask = create_mask(seq_len, mask_type='eye')
    input_tensor = th.rand(seq_len, batch_size, inp_dim)
    output_tensor = net(input_tensor, mask)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)


def metropolis_hastings_sampling(prob, start_xs, num_iters):  # mcmc sampling with transition kernel and accept ratio
    xs = start_xs.clone()
    num_sims, num_nodes = xs.shape
    device = xs.device

    sim_ids = th.arange(num_sims, device=device)

    count = 0
    for _ in range(num_iters * 8):
        if count >= num_sims * num_iters:
            break

        ids = th.randint(low=0, high=num_nodes, size=(num_sims,), device=device)
        chosen_p = prob[ids]
        chosen_xs = xs[sim_ids, ids]
        chosen_ps = th.where(chosen_xs, chosen_p, 1 - chosen_p)

        accept_rates = (1 - chosen_ps) / chosen_ps
        accept_masks = th.rand(num_sims, device=device).lt(accept_rates)
        xs[sim_ids, ids] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

        count += accept_masks.sum()
    return xs


'''run'''


def valid_in_single_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    sim_name = 'gset_14'

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 32
    show_gap = 4
    mid_dim = 2 ** 8
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        mid_dim = 2 ** 6
        num_sims = 2 ** 2
        num_repeat = 2 ** 3
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''addition'''
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    for i in range(256):
        ids_ary = th.randperm(num_nodes, device=device)[None, :]
        probs0 = th.rand(size=(1, num_nodes), device=device) * 0.02 + 0.49
        probs1 = net.auto_regression(probs0, ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=probs1[0], start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs1, 1 - probs1)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v.item(), x=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs1 * th.log2(probs1), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            temp_xs[0, :] = evaluator.best_x
            temp_xs[1:] = sim.generate_xs_randomly(num_sims=num_sims - 1)

            th.save(net.state_dict(), save_path)


def train_in_graph_distribution():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][1]
    sim_name = f'{graph_type}_{num_nodes}'
    graph_net_path = f"./graph_net_{graph_type}_Node{num_nodes}.pth"
    policy_net_path = f"./policy_trs_{graph_type}_Node{num_nodes}.pth"

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 16
    show_gap = 4
    mid_dim = 2 ** 8
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        num_sims = 2 ** 2
        num_repeat = 2 ** 3

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    assert num_nodes == sim.num_nodes

    '''addition'''
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    '''model'''
    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(graph_net_path, map_location=device))

    # policy_net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    # policy_net = PolicyRNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
    #                        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    dec_items = graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
    dec_seq, dec_matrix, dec_node = [t.detach() for t in dec_items]

    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
    for i in range(256):
        prob = prob.detach()
        '''MLP'''
        # ids_ary = th.randperm(num_nodes, device=device)[None, :]
        # probs = policy_net.auto_regression(prob[None, :], ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)
        '''TRS/RNN'''
        prob = policy_net.auto_regression(prob, dec_node, dec_matrix).clip(1e-9, 1 - 1e-9)
        probs = prob[None, :]

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=probs[0], start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs, 1 - probs)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v.item(), x=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs * th.log2(probs), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=False)

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            th.save(policy_net.state_dict(), policy_net_path)

            sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)

            solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)
            xs = sim.generate_xs_randomly(num_sims=num_sims)
            solver.reset(xs)
            for _ in tqdm.trange(8, ascii=True):
                solver.random_search(num_iters=8)

            temp_xs = solver.good_xs
            temp_vs = solver.good_vs
            evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes,
                                  x=temp_xs[0], v=temp_vs[0].item())

            prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
            dec_items = graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
            dec_seq, dec_matrix, dec_node = [t.detach() for t in dec_items]

    th.save(policy_net.state_dict(), policy_net_path)


def valid_in_graph_distribution():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    excel_id = gpu_id + 212  # todo num_nodes==600
    excel_id = gpu_id + 152  # todo num_nodes==400
    from graph_load_from_gurobi_results import load_graph_info_from_data_dir

    '''config:simulator'''
    graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][2]
    '''simulator'''
    csv_path = f"./data/syn_{graph_type}_3600.csv"
    csv_id = excel_id - 2

    txt_path, sim_name = load_graph_info_from_data_dir(csv_path, csv_id)
    graph_list = load_graph_list_from_txt(txt_path=txt_path)
    sim = SimulatorGraphMaxCut(sim_name=sim_name, graph_list=graph_list, device=device)
    num_nodes = sim.num_nodes

    '''config'''
    graph_net_path = f"./graph_net_{graph_type}_Node{num_nodes}.pth"
    policy_net_path = f"./policy_net_{graph_type}_Node{num_nodes}.pth"
    sim_name = f'{graph_type}_{num_nodes}'

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 16
    show_gap = 4
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        num_sims = 2 ** 2
        num_repeat = 2 ** 3
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)
    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 2 ** 8
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    '''model:graph_net'''
    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(graph_net_path, map_location=device))

    dec_items = graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
    dec_seq, dec_matrix, dec_node = [t.detach() for t in dec_items]

    '''model:policy_net'''
    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    # policy_net.load_state_dict(th.load(policy_net_path, map_location=device)) # todo
    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
    for i in range(256):
        prob = prob.detach()
        '''MLP'''
        # ids_ary = th.randperm(num_nodes, device=device)[None, :]
        # probs = policy_net.auto_regression(prob[None, :], ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)
        '''TRS'''
        prob = policy_net.auto_regression(prob, dec_node, dec_matrix).clip(1e-9, 1 - 1e-9)
        probs = prob[None, :]

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=prob, start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs, 1 - probs)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v.item(), x=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs * th.log2(probs), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=False)  # todo

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
    th.save(policy_net.state_dict(), save_path)


def valid_in_graph_distribution_time_limit(excel_id=2):
    from graph_load_from_gurobi_results import load_graph_info_from_data_dir

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''config:simulator'''
    graph_type = ['ErdosRenyi', 'PowerLaw', 'BarabasiAlbert'][2]
    '''simulator'''
    csv_path = f"./data/syn_{graph_type}_3600.csv"
    csv_id = excel_id - 2

    txt_path, sim_name = load_graph_info_from_data_dir(csv_path, csv_id)
    graph_list = load_graph_list_from_txt(txt_path=txt_path)
    sim = SimulatorGraphMaxCut(sim_name=sim_name, graph_list=graph_list, device=device)
    num_nodes = sim.num_nodes

    '''config'''
    graph_net_path = f"./graph_net_{graph_type}_Node{num_nodes}.pth"
    policy_net_path = f"./policy_net_{graph_type}_Node{num_nodes}.pth"
    sim_name = f'{graph_type}_{num_nodes}'

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 16
    show_gap = 4
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        num_sims = 2 ** 2
        num_repeat = 2 ** 3
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)
    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    # for _ in tqdm.trange(8, ascii=True):
    for _ in range(8):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 2 ** 8
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    '''model:graph_net'''
    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(graph_net_path, map_location=device))

    dec_items = graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
    dec_seq, dec_matrix, dec_node = [t.detach() for t in dec_items]

    '''model:policy_net'''
    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    # policy_net.load_state_dict(th.load(policy_net_path, map_location=device)) # todo
    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    warm_up_obj_value = evaluator.best_v
    exec_time_limits = list(range(10, 110 + 1, 10))  # todo 加上 warm up 的10秒钟，所以这里从10秒开始，其实是从20秒开始
    exec_obj_values = ['', ] * len(exec_time_limits)

    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
    for i in range(256):
        prob = prob.detach()
        '''MLP'''
        # ids_ary = th.randperm(num_nodes, device=device)[None, :]
        # probs = policy_net.auto_regression(prob[None, :], ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)
        '''TRS'''
        prob = policy_net.auto_regression(prob, dec_node, dec_matrix).clip(1e-9, 1 - 1e-9)
        probs = prob[None, :]

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=prob, start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs, 1 - probs)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v.item(), x=good_x)

        # if (i + 1) % show_gap == 0 or if_show_x:
        #     entropy = -th.mean(probs * th.log2(probs), dim=1)
        #     obj_entropy = entropy.mean()
        #
        #     show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
        #                 f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
        #     evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=False)  # todo

        if (i + 1) % reset_gap == 0:
            # print(f"| reset {show_gpu_memory(device=device)}")
            prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49

        exec_time = time.time() - evaluator.start_timer
        for exec_id in range(len(exec_obj_values)):
            exec_time_limit = exec_time_limits[exec_id]
            exec_obj_value = exec_obj_values[exec_id]
            if not exec_obj_value and exec_time > exec_time_limit:
                exec_obj_values[exec_id] = str(evaluator.best_v)

        if exec_obj_values[-1]:
            break

    x_str = evaluator.encoder_base64.bool_to_str(evaluator.best_x)
    x_str = x_str[1:] if x_str.startswith('\n') else x_str
    exec_obj_values_str = '  '.join(exec_obj_values)
    print(f"{warm_up_obj_value}  {exec_obj_values_str}  {x_str}")
    th.save(policy_net.state_dict(), save_path)


if __name__ == '__main__':
    # valid_in_single_graph()
    # run_in_graph_distribution()

    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ExcelID = [
        62,  # num_nodes == 100
        72,  # num_nodes == 100
        82,  # num_nodes == 100

        92,  # num_nodes == 200
        102,  # num_nodes == 200
        112,  # num_nodes == 200

        122,  # num_nodes == 300
        132,  # num_nodes == 300
        142,  # num_nodes == 300

        152,  # num_nodes == 400
        162,  # num_nodes == 400
        172,  # num_nodes == 400

        182,  # num_nodes == 500
        192,  # num_nodes == 500
        202,  # num_nodes == 500

        212,  # num_nodes == 600
        222,  # num_nodes == 600
        232,  # num_nodes == 600
    ][GPU_ID]
    print(f"Excel_ID {ExcelID} +10")
    for ParamExcelID in range(ExcelID, ExcelID + 10):
        valid_in_graph_distribution_time_limit(excel_id=ParamExcelID)

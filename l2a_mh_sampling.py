import os
import sys

import torch as th
import torch.nn as nn
import tqdm
from torch.nn.utils import clip_grad_norm_

from simulator import SimulatorGraphMaxCut
from local_search import TrickLocalSearch, show_gpu_memory
from graph_dist_trs import GraphTRS, PolicyTRS
from evaluator import Evaluator


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


def metropolis_hastings_sampling(prob, start_soluctions, num_iters):  # mcmc sampling with transition kernel and accept ratio
    xs = start_soluctions.clone()
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


def run_in_single_graph():
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
    solver = TrickLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_solutions_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_solutions
    temp_vs = solver.good_objs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, solution=temp_xs[0], obj_value=temp_vs[0].item())

    '''model'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    # net = PolicyGNN(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=mid_dim, adj_matrix=sim.adjacency_matrix).to(device)
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
        xs = metropolis_hastings_sampling(prob=probs1[0], start_soluctions=start_xs, num_iters=int(num_nodes * 0.2))
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
        if_show_x = evaluator.record2(i=i, obj_value=good_v, solution=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs1 * th.log2(probs1), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(solution=good_x, obj_value=good_v, show_str=show_str, if_show_solution=if_show_x)

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            temp_xs[0, :] = evaluator.best_x
            temp_xs[1:] = sim.generate_solutions_randomly(num_sims=num_sims - 1)

            th.save(net.state_dict(), save_path)


def run_in_graph_distribution():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = ['erdos_renyi', 'powerlaw', 'barabasi_albert'][1]
    net_path = f"./attention_net_{graph_type}_Node{num_nodes}.pth"
    sim_name = f'{graph_type}_{num_nodes}'

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
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''addition'''
    solver = TrickLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_solutions_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_solutions
    temp_vs = solver.good_objs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, solution=temp_xs[0], obj_value=temp_vs[0].item())

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    '''model'''
    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(net_path, map_location=device))
    dec_seq, dec_matrix, dec_node = [
        t.detach()
        for t in graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
    ]

    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    # prob = th.zeros(num_nodes, dtype=th.float32, device=device)
    # prob = policy_net.auto_regression(prob, dec_node, dec_matrix)

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49
    for i in range(256):
        prob = prob.detach()
        prob = policy_net.auto_regression(prob, dec_node, dec_matrix).clip(1e-9, 1 - 1e-9)
        probs1 = prob[None, :]

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=probs1[0], start_soluctions=start_xs, num_iters=int(num_nodes * 0.2))
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
        if_show_x = evaluator.record2(i=i, obj_value=good_v, solution=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs1 * th.log2(probs1), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(solution=good_x, obj_value=good_v, show_str=show_str, if_show_solution=False)  # todo

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)

            dec_seq, dec_matrix, dec_node = [
                t.detach()
                for t in graph_net.get_attn_matrix(sim.adjacency_matrix.eq(1).float()[:, None, :], mask=None)
            ]

            th.save(policy_net.state_dict(), save_path)

            solver = TrickLocalSearch(simulator=sim, num_nodes=num_nodes)

            xs = sim.generate_solutions_randomly(num_sims=num_sims)
            solver.reset(xs)
            for _ in tqdm.trange(8, ascii=True):
                solver.random_search(num_iters=8)
            temp_xs = solver.good_solutions
            temp_vs = solver.good_objs
            evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes,
                                  solution=temp_xs[0], obj_value=temp_vs[0].item())

            prob = th.rand(size=(num_nodes,), device=device) * 0.02 + 0.49


if __name__ == '__main__':
    run_in_single_graph()
    # run_in_graph_distribution()

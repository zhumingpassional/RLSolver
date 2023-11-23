import os
from simulator import SimulatorAutoregressive
from trick_local_search import *
from util import EncoderBase64
from net import PolicyMLP, Net


def train_embedding_net(sim: SimulatorAutoregressive, net_path='embedding_net.pth'):
    num_nodes = sim.num_nodes
    device = sim.device
    lr = 1e-3

    if os.path.exists(net_path):
        net = th.load(net_path, map_location=device)
        print(f"load embedding_net from {net_path}")
    else:
        adj_matrix = sim.adjacency_matrix.clone()
        adj_matrix = th.triu(adj_matrix, diagonal=0) + th.triu(adj_matrix, diagonal=1).t()
        adj_matrix = adj_matrix.ne(-1)
        num_nodes_ary = adj_matrix.sum(dim=1)
        embedding_dim = num_nodes_ary.max().item()

        net = Net(num_nodes=num_nodes, embedding_dim=embedding_dim).to(device)
        optimizer = th.optim.Adam(net.prameters(), lr=lr, maximize=False)

        ys = adj_matrix.float()
        for i in range(2 ** 12):
            xs = net(ids=th.arange(num_nodes, device=device))

            rand_is = th.randperm(num_nodes, device=device)
            xs_diff = th.pow(xs - xs[rand_is], 2).mean(dim=1)
            ys_diff = (th.pow(ys - ys[rand_is], 2).mean(dim=1) * ys.sum(dim=1)).detach()

            objective = th.pow(xs_diff - ys_diff, 2).mean()

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net.parameters(), 2)
            optimizer.step()

            if (i + 1) % (2 ** 10) == 0:
                print(f"{i:6}  {objective:9.3e}")
        th.save(net, net_path)
    return net



def roll_out_continuous(temp_xs, num_roll_continuous, rand_id_ary, num_sims, policy_net, embedding_net, dist_class,
                        sim, sim_ids, repeat_size, device):
    temp_xs = temp_xs.clone()
    th.set_grad_enabled(False)
    vs_list = []
    for k in range(32):
        xs = temp_xs.clone()
        logprobs = th.zeros(num_sims, device=device)
        for i in range(num_roll_continuous):
            ids = rand_id_ary[i].repeat(num_sims)
            ps = policy_net(xs=xs.clone(), embedding_ws=embedding_net(ids).detach())
            dist = dist_class(total_count=1, probs=ps)
            samples = dist.sample(repeat_size)[0]
            xs[sim_ids, ids] = samples

            logprobs += dist.log_prob(samples)

        vs = sim.calculate_obj_values(xs.bool()).float()
        vs_list.append(vs)
    vs = th.stack(vs_list, dim=1).float().mean(dim=1)
    th.set_grad_enabled(True)
    return vs


def run(graph_name):
    # gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gpu_id=0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    # sim_name = 'gset_14'
    sim_name = graph_name
    # x_str = X_G14
    x_str = """yNpHTLH7e2OIdP6rCrMPIFDIuNjekuOTSIcsZHJ4oVznK_DN98AUJKV9cN3W3PSVLS$h4eoCIzHrCBcGhMSuL4JD3JTg89BkvDZXVY0dh6z9NPO5IWjRxCyC
FUAYMjofiS5er"""  # 3023
    lr = 1e-5
    num_sims = 2 ** 10
    num_roll = 256
    if os.name == 'nt':  # windowsOS (new type)
        num_sims = 2 ** 4
        num_roll = 5

    sim = SimulatorAutoregressive(sim_name=sim_name, device=device)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    num_nodes = sim.num_nodes

    best_x = enc.str_to_bool(x_str).to(device)
    # best_x = sim.generate_xs_randomly(num_sims=1)[0]
    best_v = sim.calculate_obj_values(best_x[None, :])[0]
    print(f"{sim_name:32}  num_nodes {sim.num_nodes:4}  obj_value {best_v.item()}  ")

    '''init'''
    embedding_net = train_embedding_net(sim=sim)
    policy_net = PolicyMLP(node_dim=num_nodes, mid_dim=2 ** 8, embedding_dim=embedding_net.embedding_dim).to(device)
    optimizer = th.optim.Adam(policy_net.parameters(), lr=lr, maximize=True)

    from torch.distributions import Binomial
    xs = th.zeros((num_sims, num_nodes), dtype=th.float32, device=device) + 0.5
    rand_id_ary = th.randperm(num_nodes, device=device)
    sim_ids = th.arange(num_sims, device=device)
    repeat_size = th.Size((1,))

    if num_roll > num_nodes:
        print(f"change num_roll {num_roll} to num_nodes {num_nodes}")
        num_roll = num_nodes
    for j in range(2 ** 10):
        th.set_grad_enabled(True)
        logprobs = th.zeros(num_sims, device=device)
        for i in range(num_roll):
            ids = rand_id_ary[i].repeat(num_sims)
            ps = policy_net(xs=xs.clone(), embedding_ws=embedding_net(ids).detach())
            dist = Binomial(total_count=1, probs=ps)
            samples = dist.sample(repeat_size)[0]
            xs[sim_ids, ids] = samples

            logprobs += dist.log_prob(samples)

        temp_xs = xs.clone()
        num_roll_continuous = num_nodes - num_roll
        dist_class = Binomial
        vs = roll_out_continuous(temp_xs, num_roll_continuous, rand_id_ary,
                                 num_sims, policy_net, embedding_net, dist_class,
                                 sim, sim_ids, repeat_size, device)

        advantages = (vs - vs.mean()) / (vs.std() + 1e-6).detach()
        logprobs = logprobs - logprobs.mean().detach()

        objective = (advantages * logprobs.exp().clip(1e-20, 1e20)).mean()

        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(policy_net.parameters(), 2)
        optimizer.step()
        print(f"{j:6}  {objective:9.2e}  {vs.mean():9.2f}  {vs.max():6}")
    print()


if __name__ == '__main__':
    graph_name='gset_15'
    run(graph_name)
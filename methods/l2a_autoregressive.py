from simulator import MaxcutSimulatorAutoregressive
from local_search import *
from l2a_evaluator import EncoderBase64
from l2a_net import PolicyMLP, Net
from torch.distributions import Binomial

from methods.config import *

def train_embedding_net(simulator: MaxcutSimulatorAutoregressive, net_path='embedding_net.pth'):
    num_nodes = simulator.num_nodes
    device = simulator.device
    lr = 1e-3

    if os.path.exists(net_path):
        net = th.load(net_path, map_location=device)
        print(f"load embedding_net from {net_path}")
    else:
        adj_matrix = simulator.adjacency_matrix.clone()
        adj_matrix = th.triu(adj_matrix, diagonal=0) + th.triu(adj_matrix, diagonal=1).t()
        adj_matrix = adj_matrix.ne(-1)
        num_nodes_ary = adj_matrix.sum(dim=1)
        embedding_dim = num_nodes_ary.max().item()

        net = Net(num_nodes=num_nodes, embedding_dim=embedding_dim).to(device)
        optimizer = th.optim.Adam(net.prameters(), lr=lr, maximize=False)

        adjs = adj_matrix.float()
        for i in range(2 ** 12):
            solutions = net(ids=th.arange(num_nodes, device=device))

            rand_is = th.randperm(num_nodes, device=device)
            solutions_diff = th.pow(solutions - solutions[rand_is], 2).mean(dim=1)
            adjs_diff = (th.pow(adjs - adjs[rand_is], 2).mean(dim=1) * adjs.sum(dim=1)).detach()

            objective = th.pow(solutions_diff - adjs_diff, 2).mean()

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net.parameters(), 2)
            optimizer.step()

            if (i + 1) % (2 ** 10) == 0:
                print(f"{i:6}  {objective:9.3e}")
        th.save(net, net_path)
    return net


def roll_out_continuous(temp_objs, num_roll_continuous, rand_id_ary, num_simulators, policy_net, embedding_net, distri_class,
                        simulator, simulator_ids, repeat_size, device):
    temp_objs = temp_objs.clone()
    th.set_grad_enabled(False)
    objs_list = []
    for k in range(32):
        objs = temp_objs.clone()
        logprobs = th.zeros(num_simulators, device=device)
        for i in range(num_roll_continuous):
            ids = rand_id_ary[i].repeat(num_simulators)
            probs = policy_net(solutions=objs.clone(), embedding_ws=embedding_net(ids).detach())
            distri = distri_class(total_count=1, probs=probs)
            samples = distri.sample(repeat_size)[0]
            objs[simulator_ids, ids] = samples

            logprobs += distri.log_prob(samples)

        objs = simulator.calculate_obj_values(objs.bool()).float()
        objs_list.append(objs)
    objs = th.stack(objs_list, dim=1).float().mean(dim=1)
    th.set_grad_enabled(True)
    return objs


def run(graph_name):
    device = DEVICE
    x_str = """yNpHTLH7e2OIdP6rCrMPIFDIuNjekuOTSIcsZHJ4oVznK_DN98AUJKV9cN3W3PSVLS$h4eoCIzHrCBcGhMSuL4JD3JTg89BkvDZXVY0dh6z9NPO5IWjRxCyC
FUAYMjofiS5er"""  # 3023
    lr = 1e-5
    num_simulators = 2 ** 10
    num_rolls = 256
    if os.name == 'nt':  # windowsOS (new type)
        num_simulators = 2 ** 4
        num_rolls = 5

    simulator = MaxcutSimulatorAutoregressive(graph_name=graph_name, device=device)
    enc = EncoderBase64(num_nodes=simulator.num_nodes)
    num_nodes = simulator.num_nodes

    best_solution = enc.str_to_bool(x_str).to(device)
    best_obj = simulator.calculate_obj_values(best_solution[None, :])[0]
    print(f"{graph_name:32}  num_nodes {simulator.num_nodes:4}  obj_value {best_obj.item()}  ")

    '''init'''
    embedding_net = train_embedding_net(simulator=simulator)
    policy_net = PolicyMLP(node_dim=num_nodes, mid_dim=2 ** 8, embedding_dim=embedding_net.embedding_dim).to(device)
    optimizer = th.optim.Adam(policy_net.parameters(), lr=lr, maximize=True)

    solutions = th.zeros((num_simulators, num_nodes), dtype=th.float32, device=device) + 0.5
    rand_id_ary = th.randperm(num_nodes, device=device)
    simulator_ids = th.arange(num_simulators, device=device)
    repeat_size = th.Size((1,))

    if num_rolls > num_nodes:
        print(f"change num_roll {num_rolls} to num_nodes {num_nodes}")
        num_rolls = num_nodes
    for j in range(2 ** 10):
        th.set_grad_enabled(True)
        logprobs = th.zeros(num_simulators, device=device)
        for i in range(num_rolls):
            ids = rand_id_ary[i].repeat(num_simulators)
            probs = policy_net(solutions=solutions.clone(), embedding_ws=embedding_net(ids).detach())
            distri = Binomial(total_count=1, probs=probs)
            samples = distri.sample(repeat_size)[0]
            solutions[simulator_ids, ids] = samples

            logprobs += distri.log_prob(samples)

        temp_solutions = solutions.clone()
        num_roll_continuous = num_nodes - num_rolls
        distri_class = Binomial
        objs = roll_out_continuous(temp_solutions, num_roll_continuous, rand_id_ary,
                                 num_simulators, policy_net, embedding_net, distri_class,
                                 simulator, simulator_ids, repeat_size, device)

        advantages = (objs - objs.mean()) / (objs.std() + 1e-6).detach()
        logprobs = logprobs - logprobs.mean().detach()

        objective = (advantages * logprobs.exp().clip(1e-20, 1e20)).mean()

        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(policy_net.parameters(), 2)
        optimizer.step()
        print(f"{j:6}  {objective:9.2e}  {objs.mean():9.2f}  {objs.max():6}")
    print()


if __name__ == '__main__':
    graph_name='gset_15'
    run(graph_name)
from trick_local_search import *
from torch.nn.utils import clip_grad_norm_
from simulator import MaxcutSimulatorReinforce
from util import load_graph, load_graph_auto
from torch.distributions.categorical import Categorical
from net import PolicyGNN
from evaluator import Evaluator2
from config import *
def map_to_power_of_two(x):
    n = 0
    while 2 ** n <= x:
        n += 1
    return 2 ** (n - 1)


def train_embedding_net(adjacency_matrix, num_embed: int, num_epoch: int = 2 ** 10):
    num_nodes = adjacency_matrix.shape[0]
    assert num_nodes == adjacency_matrix.shape[1]
    lr = 4e-3

    '''network'''
    encoder = nn.Sequential(nn.Linear(num_nodes, num_nodes), nn.ReLU(), nn.BatchNorm1d(num_nodes),
                            nn.Linear(num_nodes, num_embed), nn.ReLU(), nn.BatchNorm1d(num_embed), )
    decoder = nn.Sequential(nn.Linear(num_embed, num_nodes), nn.ReLU(), nn.BatchNorm1d(num_nodes),
                            nn.Linear(num_nodes, num_nodes), )
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = th.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    '''train loop'''
    device = adjacency_matrix.device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    inp = adjacency_matrix

    optimizer.param_groups[0]['lr'] = lr
    for i in range(1, num_epoch + 1):
        mid = encoder(inp)
        out = decoder(mid)
        loss = criterion(inp, out)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(params, 1)
        optimizer.step()

        if not (i % 512):
            print(
                f"|lr {lr:9.3f}  {i:8}  obj*num_nodes {loss * num_nodes:9.3f}  mid {mid.min():9.2f} < {mid.max():9.2f}")
    del decoder

    encoder.eval()
    return encoder


def build_input_tensor(solutions, sim: MaxcutSimulatorReinforce, inp_dim, feature):
    num_sims, num_nodes = solutions.shape

    input = th.empty((num_sims, num_nodes, inp_dim), dtype=th.float32, device=solutions.device)
    input[:, :, 0] = solutions
    input[:, :, 1] = sim.calculate_obj_values_for_loop(solutions, if_sum=False)
    input[:, :, 2] = sim.n0_num_n1
    input[:, :, 3:] = feature
    return input


def check_gnn():
    graph_name, num_nodes = 'gset_14', 800
    inp_dim = map_to_power_of_two(num_nodes) // 2
    mid_dim = 64
    out_dim = 1
    num_embed = inp_dim - 3  # map_to_power_of_two(num_nodes) // 2

    num_sims = 8

    device = DEVICE
    graph, _, _ = load_graph(graph_name=graph_name)

    sim = MaxcutSimulatorReinforce(graph=graph, device=device, if_bidirectional=True)

    '''get adjacency_feature'''
    embed_net_path = f"{graph_name}_embedding_net.pth"
    if os.path.exists(embed_net_path):
        embed_net = th.load(embed_net_path, map_location=device)
        embed_net.eval()
    else:
        embed_net = train_embedding_net(adjacency_matrix=sim.adjacency_matrix, num_embed=num_embed, num_epoch=2 ** 14)
        th.save(embed_net, embed_net_path)
    sim.adjacency_feature = embed_net(sim.adjacency_matrix).detach()

    '''build net'''
    net = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim).to(device)
    print(f"num_nodes {num_nodes}  num_node_features {sim.adjacency_feature.shape[1]}")

    xs = sim.generate_solutions_randomly(num_sims=num_sims)
    inp = build_input_tensor(solutions=xs, sim=sim, inp_dim=inp_dim, feature=sim.adjacency_feature)
    out = net(inp, sim.adjacency_indies)
    print(out.shape)


def search_and_evaluate_reinforce(graph_name='gset_14', num_nodes=800, gpu_id=0):
    inp_dim = map_to_power_of_two(num_nodes) // 2
    mid_dim = 128
    out_dim = 1
    num_embed = inp_dim - 3  # map_to_power_of_two(num_nodes) // 2

    num_sims = 2 ** 9
    num_reset = 2 ** 0
    num_iter1 = 2 ** 6
    num_iter0 = 2 ** 1

    if os.name == 'nt':
        num_sims = 2 ** 3
        num_reset = 2 ** 1
        num_iter1 = 2 ** 4
        num_iter0 = 2 ** 2

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    '''build simulator'''
    device = DEVICE
    graph, _, _ = load_graph_auto(graph_name=graph_name)

    simulator = MaxcutSimulatorReinforce(graph=graph, device=device, if_bidirectional=True)

    '''get adjacency_feature'''
    embed_net_path = f"{graph_name}_embedding_net.pth"
    if os.path.exists(embed_net_path):
        embed_net = th.load(embed_net_path, map_location=device)
        embed_net.eval()
    else:
        embed_net = train_embedding_net(adjacency_matrix=simulator.adjacency_matrix, num_embed=num_embed, num_epoch=2 ** 14)
        th.save(embed_net, embed_net_path)
    simulator.adjacency_feature = embed_net(simulator.adjacency_matrix).detach()

    '''build net'''
    net = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim).to(device)
    net_params = list(net.parameters())
    print(f"num_nodes {num_nodes}  num_node_features {simulator.adjacency_feature.shape[1]}")

    trick = TrickLocalSearch(simulator=simulator, num_nodes=num_nodes)
    optimizer = th.optim.Adam(net_params, lr=2e-3, maximize=True)

    '''evaluator'''
    temp_xs = simulator.generate_solutions_randomly(num_sims=1)
    temp_vs = simulator.calculate_obj_values(temp_xs)
    evaluator = Evaluator2(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, solution=temp_xs[0], obj=temp_vs[0].item())
    del temp_xs, temp_vs

    print("start searching")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(1, num_reset + 1):
        prev_solutions = simulator.generate_solutions_randomly(num_sims)
        prev_objs = simulator.calculate_obj_values(prev_solutions)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_objs.argmax()
            solutions = prev_solutions[prev_i:prev_i + 1].repeat(num_sims, 1)
            objs = prev_objs[prev_i:prev_i + 1].repeat(num_sims)

            '''update xs via probability, obtain logprobs for VPG'''
            logprobs = th.empty((num_sims, num_iter0), dtype=th.float32, device=device)
            for i0 in range(num_iter0):
                if i0 == 0:
                    output_tensor = th.ones((num_sims, num_nodes), dtype=th.float32, device=device) / num_nodes
                else:
                    input_tensor = build_input_tensor(solutions=solutions.clone().detach(), sim=simulator, inp_dim=inp_dim,
                                                      feature=simulator.adjacency_feature.detach())
                    output_tensor = net(input_tensor, simulator.adjacency_indies)
                distri = Categorical(probs=output_tensor)
                sample = distri.sample(th.Size((1,)))[0]
                solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])

                logprobs[:, i0] = distri.log_prob(sample)
            logprobs = logprobs.mean(dim=1)
            logprobs = logprobs - logprobs.mean()

            '''update xs via max local search'''
            trick.reset(solutions)
            trick.random_search(num_iters=2 ** 6, num_spin=8, noise_std=0.2)
            advantage_value = (trick.good_objs - objs).detach()

            objective = (logprobs.exp() * advantage_value).mean()

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            prev_solutions = trick.good_solutions.clone()
            prev_objs = trick.good_objs.clone()

            if j1 > num_skip and j1 % gap_print == 0:
                good_i = trick.good_objs.argmax()
                i = j2 * num_iter1 + j1
                solution = trick.good_solutions[good_i]
                obj = trick.good_objs[good_i].item()

                evaluator.record2(i=i, obj=obj, solution=solution)
                evaluator.logging_print(obj=obj)


if __name__ == '__main__':
    search_and_evaluate_reinforce()
    # check_gnn()

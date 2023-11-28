import os
import sys
import time
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from simulator import MaxcutSimulator, MaxcutSimulatorAutoregressive
from evaluator import X_G14, X_G15, X_G49, X_G50, X_G22, X_G55, X_G70
from evaluator import Evaluator, Evaluator2
from util import EncoderBase64, calc_device
TEN = th.Tensor

'''local search'''


class TrickLocalSearch:
    def __init__(self, simulator: MaxcutSimulator, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_solutions = th.tensor([])  # solution x
        self.good_objs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        self.good_solutions = xs
        self.good_objs = self.simulator.calculate_obj_values(xs=xs)
        self.num_sims = xs.shape[0]

    # def reset_search(self, num_sims):
    #     solutions = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
    #     for sim_id in range(num_sims):
    #         _solutions = self.simulator.generate_solutions_randomly(num_sims=num_sims)
    #         _objs = self.simulator.calculate_obj_values(_solutions)
    #         solutions[sim_id] = _solutions[_objs.argmax()]
    #     return solutions

    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_solutions = self.good_solutions.clone()
        prev_objs_raw = sim.calculate_obj_values_for_loop(prev_solutions, if_sum=False)
        prev_objs = prev_objs_raw.sum(dim=1)

        thresh = None
        for _ in range(num_iters):
            '''flip randomly with ws(weights)'''
            ws = sim.n0_num_n1 - (4 if sim.if_bidirectional else 2) * prev_objs_raw
            ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]

            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * (ws_std.float() * noise_std)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)

            solutions = prev_solutions.clone()
            solutions[spin_mask] = th.logical_not(solutions[spin_mask])
            objs = sim.calculate_obj_values(solutions)

            update_solutions_by_objs(prev_solutions, prev_objs, solutions, objs)

        '''addition'''
        for i in range(sim.num_nodes):
            solutions1 = prev_solutions.clone()
            solutions1[:, i] = th.logical_not(solutions1[:, i])
            objs1 = sim.calculate_obj_values(solutions1)

            update_solutions_by_objs(prev_solutions, prev_objs, solutions1, objs1)

        update_solutions_by_objs(self.good_solutions, self.good_objs, prev_solutions, prev_objs)
        return self.good_solutions, self.good_objs


def update_solutions_by_objs(solutions0, objs0, solutions1, objs1):
    good_is = objs1.gt(objs0)
    solutions0[good_is] = solutions1[good_is]
    objs0[good_is] = objs1[good_is]


'''network'''


class PolicyMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, out_dim), nn.Tanh(), )
        self.net2 = nn.Sequential(nn.Linear(1 + out_dim // inp_dim, 4), nn.Tanh(),
                                  nn.Linear(4, 1), nn.Sigmoid(), )

    def forward(self, solutions0):
        num_simulators, num_nodes = solutions0.shape
        solutions1 = self.net1(solutions0).reshape((num_simulators, num_nodes, -1))
        solutions2 = th.cat((solutions0.unsqueeze(2), solutions1), dim=2)
        solutions3 = self.net2(solutions2).squeeze(2)
        return solutions3


def train_loop(num_train, device, seq_len, best_solution, num_simulators, simulator, net, optimizer, show_gap, noise_std):
    num_nodes = best_solution.shape[0]
    sim_ids = th.arange(num_simulators, device=simulator.device)
    start_time = time.time()
    assert seq_len <= num_nodes

    for j in range(num_train):
        mask = th.zeros(num_nodes, dtype=th.bool, device=device)
        n_std = (num_nodes - seq_len - 1) // 4
        n_avg = seq_len + 1 + n_std * 2
        rand_n = int(th.randn(size=(1,)).clip(-2, +2).item() * n_std + n_avg)
        mask[:rand_n] = True
        mask = mask[th.randperm(num_nodes)]
        rand_solution = best_solution.clone()
        rand_solution[mask] = th.logical_not(rand_solution[mask])
        rand_obj = simulator.calculate_obj_values(rand_solution[None, :])[0]
        good_solutions = rand_solution.repeat(num_simulators, 1)
        good_objs = rand_obj.repeat(num_simulators, )

        solutions = good_solutions.clone()
        num_not_equal = solutions[0].ne(best_solution).sum().item()
        # assert num_not_equal == rand_n
        # assert num_not_equal >= seq_len

        out_list = th.empty((num_simulators, seq_len), dtype=th.float32, device=device)
        for i in range(seq_len):
            net.train()
            inp = solutions.float()
            out = net(inp) + solutions.ne(best_solution).float().detach()

            noise = th.randn_like(out) * noise_std
            sample = (out + noise).argmax(dim=1)
            solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])
            objs = simulator.calculate_obj_values(solutions)

            out_list[:, i] = out[sim_ids, sample]

            update_solutions_by_objs(good_solutions, good_objs, solutions, objs)

        good_objs = good_objs.float()
        advantage = (good_objs - good_objs.mean()) / (good_objs.std() + 1e-6)

        objective = (out_list.mean(dim=1) * advantage.detach()).mean()
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net.parameters(), 2)
        optimizer.step()

        if (j + 1) % show_gap == 0:
            objs_avg = good_objs.mean().item()
            print(f'{j:8}  {time.time() - start_time:9.0f} '
                  f'| {objs_avg:9.3f}  {objs_avg - rand_obj.item():9.3f} |  {num_not_equal}')
    pass


def check_net(net, sim, num_sims):
    num_nodes = sim.num_nodes
    good_solutions = sim.generate_solutions_randomly(num_sims=num_sims)
    good_objs = sim.calculate_obj_values(good_solutions)

    solutions = good_solutions.clone()
    sim_ids = th.arange(num_sims, device=sim.device)
    for i in range(num_nodes):
        inp = solutions.float()
        out = net(inp)

        sample = out.argmax(dim=1)
        solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])
        objs = sim.calculate_obj_values(solutions)

        update_solutions_by_objs(good_solutions, good_objs, solutions, objs)
    return good_solutions, good_objs


def check_generate_best_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # sim_name = 'gset_14'
    # x_str = X_G14
    graph_name = 'gset_70'
    x_str = X_G70
    lr = 1e-3
    noise_std = 0.1

    num_train = 2 ** 9
    mid_dim = 2 ** 8
    seq_len = 2 ** 6
    show_gap = 2 ** 5

    num_sims = 2 ** 8
    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4

    device = calc_device(gpu_id)

    '''simulator'''
    simulator = MaxcutSimulator(graph_name=graph_name, device=device)
    enc = EncoderBase64(num_nodes=simulator.num_nodes)
    num_nodes = simulator.num_nodes

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes * 3).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=True)

    best_x = enc.str_to_bool(x_str).to(device)
    best_v = simulator.calculate_obj_values(best_x[None, :])[0]
    print(f"{graph_name:32}  num_nodes {simulator.num_nodes:4}  obj_value {best_v.item()}  ")

    train_loop(num_train, device, seq_len, best_x, num_sims, simulator, net, optimizer, show_gap, noise_std)


'''run'''


def find_smallest_nth_power_of_2(target):
    n = 0
    while 2 ** n < target:
        n += 1
    return 2 ** n


def search_and_evaluate_local_search():
    gpu_id = 0

    # if_reinforce = False
    # num_reset = 2 ** 1
    # num_iter1 = 2 ** 6
    # num_iter0 = 2 ** 4
    # num_sims = 2 ** 13

    if_reinforce = True
    num_reset = 2 ** 8
    num_iter1 = 2 ** 5
    num_iter0 = 2 ** 6
    num_sims = 2 ** 11
    num_sims1 = 2 ** 10

    seq_len = 2 ** 7
    show_gap = 2 ** 6
    num_train = 2 ** 9

    noise_std = 0.1
    mid_dim = 2 ** 7
    lr = 1e-5

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    solution_str = None
    graph_name = 'gset_14'
    if gpu_id == 0:
        graph_name = 'gset_14'  # num_nodes==800
        solution_str = """yNpHTLH7e2OIdP6rCrMPIFDIONjekuOTSIcsZHJ4oVznK_DN98AUJKV9cN3W3PSVLS$h4eoCIzHrCBcGhMSuL4JD3JTg89BkvDZXVY07h6z9NPO5QWjRxCyC
FUAYMjofiS5er"""  # 3022
    if gpu_id == 1:
        graph_name = 'gset_15'  # num_nodes==800
        solution_str = """PoaFXUkt2uOnZNChgBeg8ljjVkK_2VvBmhul_GbbYmI8GQ9h6wPDKxowYppuj9MzV_pg8oQ69gXqaFOJWCaMRnaDvqUnmtTe9ua9xVe2NS5bKcazkHsW6kO7
hUH4vj0nAzi24"""
    # if gpu_id == 2:
    #     sim_name = 'gset_49'  # num_nodes==3000
    # if gpu_id == 3:
    #     sim_name = 'gset_50'  # num_nodes==3000
    if gpu_id in {2, }:
        graph_name = 'gset_22'  # num_nodes==2000
        solution_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 5
        num_iter0 = 2 ** 7
    if gpu_id in {3, }:
        graph_name = 'gset_22'  # num_nodes==2000
        solution_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 6
    if gpu_id in {4, 5}:
        graph_name = 'gset_55'  # num_nodes==5000
        solution_str = X_G55
        num_sims1 = 2 ** 9
        seq_len = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 7
    if gpu_id in {6, 7}:
        graph_name = 'gset_70'  # num_nodes==10000
        solution_str = X_G70
        num_sims1 = 2 ** 9
        seq_len = 2 ** 5
        mid_dim = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 8

    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        num_reset = 2 ** 1
        num_iter0 = 2 ** 2

    device = calc_device(gpu_id)

    '''simulator'''
    simulator = MaxcutSimulatorAutoregressive(graph_name=graph_name, device=device)
    num_nodes = simulator.num_nodes

    '''evaluator'''
    temp_solutions = simulator.generate_solutions_randomly(num_sims=1)
    temp_objs = simulator.calculate_obj_values(solutions=temp_solutions)
    evaluator = Evaluator2(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, solution=temp_solutions[0], obj=temp_objs[0].item())

    '''trick'''
    trick = TrickLocalSearch(simulator=simulator, num_nodes=simulator.num_nodes)

    '''network'''
    mid_dim = mid_dim if mid_dim else find_smallest_nth_power_of_2(num_nodes)
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=False)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {graph_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}")
        best_solutions = simulator.generate_solutions_randomly(num_sims)
        best_objs = simulator.calculate_obj_values(best_solutions)

        if (j2 == 0) and (solution_str is not None):
            _num_iter1 = 0  # skip

            evaluator.best_solution = evaluator.encoder_base64.str_to_bool(solution_str).to(device)
            evaluator.best_obj = simulator.calculate_obj_values(evaluator.best_solution[None, :])[0]
        else:
            _num_iter1 = num_iter1
        for j1 in range(_num_iter1):
            best_i = best_objs.argmax()
            best_solutions[:] = best_solutions[best_i]
            best_objs[:] = best_objs[best_i]

            '''update xs via probability'''
            solutions = best_solutions.clone()
            _num_iter0 = th.randint(int(num_iter0 * 0.75), int(num_iter0 * 1.25), size=(1,)).item()
            for _ in range(num_iter0):
                if if_reinforce and (j2 != 0):
                    best_solution = evaluator.best_solution
                    out = net(solutions.float()) + solutions.ne(best_solution[None, :]).float()
                    sample = (out + th.rand_like(out) * noise_std).argmax(dim=1)
                else:
                    sample = th.randint(num_nodes, size=(num_sims,), device=device)
                solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])

            '''update xs via local search'''
            trick.reset(solutions)
            trick.random_search(num_iters=2 ** 6, num_spin=4)

            update_solutions_by_objs(best_solutions, best_objs, trick.good_solutions, trick.good_objs)

            if j1 > num_skip and (j1 + 1) % gap_print == 0:
                i = j2 * num_iter1 + j1

                good_i = trick.good_objs.argmax()
                good_x = trick.good_solutions[good_i]
                good_v = trick.good_objs[good_i].item()

                if_show_x = evaluator.record2(i=i, obj=good_v, x=good_x)
                evaluator.logging_print(obj=good_v, if_show_x=if_show_x)

        if if_reinforce:
            best_solution = evaluator.best_solution
            best_obj = evaluator.best_obj
            evaluator.logging_print(obj=best_obj, if_show_x=True)

            train_loop(num_train, device, seq_len, best_solution, num_sims1, simulator, net, optimizer, show_gap, noise_std)

        evaluator.plot_record()


if __name__ == '__main__':
    search_and_evaluate_local_search()

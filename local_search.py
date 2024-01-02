import os
import sys
import time
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from simulator import SimulatorGraphMaxCut
from evaluator import X_G14, X_G15, X_G49, X_G50, X_G22, X_G55, X_G70
from evaluator import Evaluator, EncoderBase64

TEN = th.Tensor

'''local search'''


class TrickLocalSearch:
    def __init__(self, simulator: SimulatorGraphMaxCut, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

        self.num_sims = 0
        self.good_solutions = th.tensor([])  # solution x
        self.good_objs = th.tensor([])  # objective value

    def reset(self, solutions: TEN):
        objs = self.simulator.calculate_obj_values(solutions=solutions)

        self.good_solutions = solutions
        self.good_objs = objs
        self.num_sims = solutions.shape[0]
        return objs

    def reset_search(self, num_sims):
        solutions = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _solutions = self.simulator.generate_solutions_randomly(num_sims=num_sims)
            _objs = self.simulator.calculate_obj_values(_solutions)
            solutions[sim_id] = _solutions[_objs.argmax()]
        return solutions

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

        num_update = update_solutions_by_objs(self.good_solutions, self.good_objs, prev_solutions, prev_objs)
        return self.good_solutions, self.good_objs, num_update


def update_solutions_by_objs(src_solutions, src_objs, dst_solutions, dst_objs):
    """
    并行的子模拟器数量为 num_sims, 解x 的节点数量为 num_nodes
    solutions: 并行数量个解x,solutions.shape == (num_sims, num_nodes)
    vs: 并行数量个解x对应的 objective value. vs.shape == (num_sims, )

    更新后，将dst_solutions, dst_objs 中 objective value数值更高的解 替换到src_solution中和 dst_objs中
    如果被更新的解的数量大于0，将返回True
    """
    good_is = dst_objs.gt(src_objs)
    src_solutions[good_is] = dst_solutions[good_is]
    src_objs[good_is] = dst_objs[good_is]
    return good_is.shape[0]


'''network'''


class PolicyMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, out_dim), nn.Tanh(), )
        self.net2 = nn.Sequential(nn.Linear(1 + out_dim // inp_dim, 4), nn.Tanh(),
                                  nn.Linear(4, 1), nn.Sigmoid(), )

    def forward(self, solutions):
        num_sims, num_nodes = solutions.shape
        solutions1 = self.net1(solutions).reshape((num_sims, num_nodes, -1))
        solutions2 = th.cat((solutions.unsqueeze(2), solutions1), dim=2)
        solutions3 = self.net2(solutions2).squeeze(2)
        return solutions3


def train_loop(num_train, device, seq_len, best_solution, num_sims1, sim, net, optimizer, show_gap, noise_std):
    num_nodes = best_solution.shape[0]
    sim_ids = th.arange(num_sims1, device=sim.device)
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
        rand_obj = sim.calculate_obj_values(rand_solution[None, :])[0]
        good_solutions = rand_solution.repeat(num_sims1, 1)
        good_objs = rand_obj.repeat(num_sims1, )

        solutions = good_solutions.clone()
        num_not_equal = solutions[0].ne(best_solution).sum().item()
        # assert num_not_equal == rand_n
        # assert num_not_equal >= seq_len

        out_list = th.empty((num_sims1, seq_len), dtype=th.float32, device=device)
        for i in range(seq_len):
            net.train()
            inp = solutions.float()
            out = net(inp) + solutions.ne(best_solution).float().detach()

            noise = th.randn_like(out) * noise_std
            sample = (out + noise).argmax(dim=1)
            solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])
            vs = sim.calculate_obj_values(solutions)

            out_list[:, i] = out[sim_ids, sample]

            update_solutions_by_objs(good_solutions, good_objs, solutions, vs)

        good_objs = good_objs.float()
        advantage = (good_objs - good_objs.mean()) / (good_objs.std() + 1e-6)

        objective = (out_list.mean(dim=1) * advantage.detach()).mean()
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net.parameters(), 2)
        optimizer.step()

        if (j + 1) % show_gap == 0:
            vs_avg = good_objs.mean().item()
            print(f'{j:8}  {time.time() - start_time:9.0f} '
                  f'| {vs_avg:9.3f}  {vs_avg - rand_obj.item():9.3f} |  {num_not_equal}')
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
        vs = sim.calculate_obj_values(solutions)

        update_solutions_by_objs(good_solutions, good_objs, solutions, vs)
    return good_solutions, good_objs


def check_generate_best_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # sim_name = 'gset_14'
    # x_str = X_G14
    sim_name = 'gset_70'
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

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    num_nodes = sim.num_nodes

    '''network'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes * 3).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=True)

    best_x = enc.str_to_bool(x_str).to(device)
    best_v = sim.calculate_obj_values(best_x[None, :])[0]
    print(f"{sim_name:32}  num_nodes {sim.num_nodes:4}  obj_value {best_v.item()}  ")

    train_loop(num_train, device, seq_len, best_x, num_sims, sim, net, optimizer, show_gap, noise_std)


'''utils'''


def show_gpu_memory(device):
    if not th.cuda.is_available():
        return 'not th.cuda.is_available()'

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    show_str = (
        f"AllRAM {total_memory:.2f} GB, "
        f"MaxRAM {max_allocated:.2f} GB, "
        f"NowRAM {memory_allocated:.2f} GB, "
        f"Rate {(max_allocated / total_memory) * 100:.2f}%"
    )
    return show_str


'''run'''


def find_smallest_nth_power_of_2(target):
    n = 0
    while 2 ** n < target:
        n += 1
    return 2 ** n


def search_and_evaluate_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

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

    x_str = None
    sim_name = 'gset_14'
    if gpu_id == 0:
        sim_name = 'gset_14'  # num_nodes==800
        x_str = """
yNpHTLH7e2OIdP6rCrMPIFDIONjekuOTSIcsZHJ4oVznK_DN98AUJKV9cN3W3PSVLS$h4eoCIzHrCBcGhMSuL4JD3JTg89BkvDZXVY07h6z9NPO5QWjRxCyC
FUAYMjofiS5er"""  # 3022
    if gpu_id == 1:
        sim_name = 'gset_15'  # num_nodes==800
        x_str = """
PoaFXUkt2uOnZNChgBeg8ljjVkK_2VvBmhul_GbbYmI8GQ9h6wPDKxowYppuj9MzV_pg8oQ69gXqaFOJWCaMRnaDvqUnmtTe9ua9xVe2NS5bKcazkHsW6kO7
hUH4vj0nAzi24"""
    # if gpu_id == 2:
    #     sim_name = 'gset_49'  # num_nodes==3000
    # if gpu_id == 3:
    #     sim_name = 'gset_50'  # num_nodes==3000
    if gpu_id in {2, }:
        sim_name = 'gset_22'  # num_nodes==2000
        x_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 5
        num_iter0 = 2 ** 7
    if gpu_id in {3, }:
        sim_name = 'gset_22'  # num_nodes==2000
        x_str = X_G22
        seq_len = 2 ** 6
        num_sims1 = 2 ** 9
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 6
    if gpu_id in {4, 5}:
        sim_name = 'gset_55'  # num_nodes==5000
        x_str = X_G55
        num_sims1 = 2 ** 9
        seq_len = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 7
    if gpu_id in {6, 7}:
        sim_name = 'gset_70'  # num_nodes==10000
        x_str = X_G70
        num_sims1 = 2 ** 9
        seq_len = 2 ** 5
        mid_dim = 2 ** 6
        num_iter1 = 2 ** 6
        num_iter0 = 2 ** 8

    if os.name == 'nt':  # windows new type
        num_sims = 2 ** 4
        num_reset = 2 ** 1
        num_iter0 = 2 ** 2

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    simulator_class = SimulatorGraphMaxCut
    solver_class = TrickLocalSearch

    '''simulator'''
    sim = simulator_class(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''evaluator'''
    temp_solutions = sim.generate_solutions_randomly(num_sims=1)
    temp_objs = sim.calculate_obj_values(solutions=temp_solutions)
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_solutions[0],
                          v=temp_objs[0].item())

    '''solver'''
    solver = solver_class(simulator=sim, num_nodes=sim.num_nodes)

    '''network'''
    mid_dim = mid_dim if mid_dim else find_smallest_nth_power_of_2(num_nodes)
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr=lr, maximize=False)

    """loop"""
    th.set_grad_enabled(True)
    print(f"start searching, {sim_name}  num_nodes={num_nodes}")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(num_reset):
        print(f"|\n| reset {j2}")
        best_solutions = sim.generate_solutions_randomly(num_sims)
        best_objs = sim.calculate_obj_values(best_solutions)

        if (j2 == 0) and (x_str is not None):
            _num_iter1 = 0  # skip

            evaluator.best_x = evaluator.encoder_base64.str_to_bool(x_str).to(device)
            evaluator.best_v = sim.calculate_obj_values(evaluator.best_x[None, :])[0]
        else:
            _num_iter1 = num_iter1
        for j1 in range(_num_iter1):
            best_i = best_objs.argmax()
            best_solutions[:] = best_solutions[best_i]
            best_objs[:] = best_objs[best_i]

            '''update solutions via probability'''
            solutions = best_solutions.clone()
            for _ in range(num_iter0):
                if if_reinforce and (j2 != 0):
                    best_x = evaluator.best_x
                    out = net(solutions.float()) + solutions.ne(best_x[None, :]).float()
                    sample = (out + th.rand_like(out) * noise_std).argmax(dim=1)
                else:
                    sample = th.randint(num_nodes, size=(num_sims,), device=device)
                solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])

            # best_x = evaluator.best_x
            # '''auto-regressive'''
            # for _ in range(num_iter0):
            #     out = net(solutions.float()) + solutions.ne(best_x[None, :]).float()
            #     sample = (out + th.rand_like(out) * noise_std).argmax(dim=1)
            #     solutions[sim_ids, sample] = th.logical_not(solutions[sim_ids, sample])
            # '''directly'''
            # out = net(solutions.float()) + solutions.ne(best_x[None, :]).float()
            # sample = out >= th.kthvalue(out, k=num_iter0, dim=1)[0][:, None]
            # solutions[sample] = th.logical_not(solutions[sample])

            '''update solutions via local search'''
            solver.reset(solutions)
            solver.random_search(num_iters=2 ** 6, num_spin=4)

            update_solutions_by_objs(best_solutions, best_objs, solver.good_solutions, solver.good_objs)

            if j1 > num_skip and (j1 + 1) % gap_print == 0:
                i = j2 * num_iter1 + j1

                good_i = solver.good_objs.argmax()
                good_x = solver.good_solutions[good_i]
                good_v = solver.good_objs[good_i].item()

                if_show_x = evaluator.record2(i=i, v=good_v, x=good_x)
                evaluator.logging_print(x=good_x, v=good_v, show_str=f"{good_v:6}", if_show_x=if_show_x)

        if if_reinforce:
            best_x = evaluator.best_x
            best_v = evaluator.best_v
            evaluator.logging_print(x=best_x, v=best_v, show_str=f"{best_v:9.0f}", if_show_x=True)

            train_loop(num_train, device, seq_len, best_x, num_sims1, sim, net, optimizer, show_gap, noise_std)

        evaluator.plot_record()


if __name__ == '__main__':
    search_and_evaluate_local_search()

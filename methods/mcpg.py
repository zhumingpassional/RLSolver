import os
import torch as th
import sys
from torch_geometric.data import Data
from methods.L2A.evaluator import EncoderBase64
from methods.L2A.maxcut_simulator import load_graph_list
from envs.env_mcpg_maxcut import (metro_sampling,
                                  pick_good_xs,
                                  get_return,
                                  Sampler
                                  )

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0





def maxcut_dataloader(path, device=th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = th.LongTensor(2, num_edges)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                cnt += 1

        data = Data(num_nodes=num_nodes, edge_index=edge_index.to(device))
        data = append_neighbors(data)

        data.single_degree = []
        data.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data.num_nodes):
            data.single_degree.append(len(data.neighbors[i0]))
            data.weighted_degree.append(
                float(th.sum(data.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(th.sum(th.abs(data.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = th.tensor(tensor_abs_weighted_degree)
        data.sorted_degree_nodes = th.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = th.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(
                tensor_abs_weighted_degree[edge_index[0][i0]] + tensor_abs_weighted_degree[edge_index[1][i0]])
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = 1 - data.weighted_degree[node_r] / 2 - 0.05
            add[1][i0] = 1 - data.weighted_degree[node_c] / 2 - 0.05
            add[2][i0] = 1 + 0.05

        for i0 in range(num_nodes):
            data.neighbor_edges[i0] = data.neighbor_edges[i0].unsqueeze(0)
        data.add_items = add
        edge_degree = th.tensor(edge_degree)
        data.sorted_degree_edges = th.argsort(
            edge_degree, descending=True)
        return data, num_nodes


def append_neighbors(data, device=th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')):
    data.neighbors = []
    data.neighbor_edges = []
    # num_nodes = data.encode_len
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    edge_weight = 1
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = th.LongTensor(data.n0[index]).to(device)
        data.n1[index] = th.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = th.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = th.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)

    for i in range(num_nodes):
        data.neighbors[i] = th.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = th.tensor(
            data.neighbor_edges[i]).to(device)
    return data

def save_graph_list_to_txt(graph_list, txt_path: str):
    num_nodes = max([max(n0, n1) for n0, n1, distance in graph_list]) + 1
    num_edges = len(graph_list)

    lines = [f"{num_nodes} {num_edges}", ]
    lines.extend([f"{n0 + 1} {n1 + 1} {distance}" for n0, n1, distance in graph_list])
    lines = [l + '\n' for l in lines]
    with open(txt_path, 'w') as file:
        file.writelines(lines)


def print_gpu_memory(device):
    if not th.cuda.is_available():
        return

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    print(f"AllRAM {total_memory:.2f} GB, "
          f"MaxRAM {max_allocated:.2f} GB, "
          f"NowRAM {memory_allocated:.2f} GB, "
          f"Rate {(max_allocated / total_memory) * 100:.2f}%")


def run():
    max_epoch_num = 2 ** 13
    sample_epoch_num = 8
    repeat_times = 128

    num_ls = 8
    reset_epoch_num = 128
    total_mcmc_num = 512
    # path = 'data/gset_14.txt'
    # path = 'data/gset_15.txt'
    # path = 'data/gset_49.txt'
    # path = 'data/gset_50.txt'
    graph_type = ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw'][2]
    num_nodes = 300
    graph_id = 0
    graph_name = f"{graph_type}_{num_nodes}_ID{graph_id}"
    path = f'temp_{graph_name}.txt'
    save_graph_list_to_txt(graph_list=load_graph_list(graph_name=graph_name), txt_path=path)

    # num_ls = 6
    # reset_epoch_num = 192
    # total_mcmc_num = 224
    # path = 'data/gset_22.txt'

    # num_ls = 8
    # reset_epoch_num = 128
    # total_mcmc_num = 256
    # path = 'data/gset_55.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # total_mcmc_num = 192
    # path = 'data/gset_70.txt'

    # num_ls = 8
    # reset_epoch_num = 256
    # repeat_times = 512
    # total_mcmc_num = 2048
    # path = 'data/gset_22.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 192
    # repeat_times = 448
    # total_mcmc_num = 1024
    # path = 'data/gset_55.txt'  # GPU RAM 40GB

    # num_ls = 8
    # reset_epoch_num = 320
    # repeat_times = 288
    # total_mcmc_num = 768
    # path = 'data/gset_70.txt'  # GPU RAM 40GB

    show_gap = 2 ** 4

    if os.name == 'nt':
        max_epoch_num = 2 ** 4
        repeat_times = 32
        reset_epoch_num = 32
        total_mcmc_num = 64
        show_gap = 2 ** 0

    '''init'''
    sim_name = path  # os.path.splitext(os.path.basename(path))[0]
    data, num_nodes = maxcut_dataloader(path)
    device = th.device(f'cuda:{GPU_ID}' if th.cuda.is_available() else 'cpu')

    change_times = int(num_nodes / 10)  # transition times for metropolis sampling

    net = Sampler(num_nodes)
    net.to(device).reset()
    optimizer = th.optim.Adam(net.parameters(), lr=8e-2)

    '''addition'''
    from methods.L2A.maxcut_simulator import SimulatorMaxcut
    from methods.L2A.maxcut_local_search import SolverLocalSearch
    sim = SimulatorMaxcut(sim_name=sim_name, device=device)
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=total_mcmc_num)
    solver.reset(xs.bool())
    for _ in range(16):
        solver.random_search(num_iters=repeat_times // 16)
    now_max_info = solver.good_xs.t()
    now_max_res = solver.good_vs
    del sim
    del solver

    '''loop'''
    net.train()
    xs_prob = (th.zeros(num_nodes) + 0.5).to(device)
    xs_bool = now_max_info.repeat(1, repeat_times)

    print('start loop')
    sys.stdout.flush()  # add for slurm stdout
    for epoch in range(1, max_epoch_num + 1):
        net.to(device).reset()

        for j1 in range(reset_epoch_num // sample_epoch_num):
            xs_sample = metro_sampling(xs_prob, xs_bool.clone(), change_times)

            temp_max, temp_max_info, value = pick_good_xs(
                data, xs_sample, num_ls, total_mcmc_num, repeat_times, device)
            # update now_max
            for i0 in range(total_mcmc_num):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = th.argmax(now_max_res)
            now_min_index = th.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            xs_bool = temp_max_info.clone()
            xs_bool = xs_bool.repeat(1, repeat_times)
            # construct the start point for next iteration
            start_samples = xs_sample.t()

            probs = xs_prob[None, :]
            _probs = 1 - probs
            entropy = -(probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()

            print(f"value {max(now_max_res).item():9.2f}  entropy {obj_entropy:9.3f}")
            sys.stdout.flush()  # add for slurm stdout

            for _ in range(sample_epoch_num):
                xs_prob = net()
                ret_loss_ls = get_return(xs_prob, start_samples, value, total_mcmc_num, repeat_times)

                optimizer.zero_grad()
                ret_loss_ls.backward()
                th.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()
            th.cuda.empty_cache()

            if j1 % show_gap == 0:
                total_max = now_max_res
                best_sort = th.argsort(now_max_res, descending=True)
                total_best_info = th.squeeze(now_max_info[:, best_sort[0]])

                objective_value = max(total_max)
                solution = total_best_info

                encoder = EncoderBase64(encode_len=num_nodes)
                x_str = encoder.bool_to_str(x_bool=solution)

                print(f"epoch {epoch:6}  value {objective_value.item():8.2f}  {x_str}")
                print_gpu_memory(device)

            if os.path.exists('./stop'):
                break
        if os.path.exists('./stop'):
            break
    if os.path.exists('./stop'):
        print(f"break: os.path.exists('./stop') {os.path.exists('./stop')}")
        sys.stdout.flush()  # add for slurm stdout


if __name__ == '__main__':
    run()

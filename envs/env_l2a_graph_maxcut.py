import os
import sys
import logging
import torch as th
from torch.nn.utils import clip_grad_norm_

from config import ConfigPolicy, ConfigGraph, GraphList
from baseline.l2a_network import reset_parameters_of_model, GraphTRS
from baseline.l2a_evaluator import Evaluator, read_info_from_recorder
from baseline.l2a_graph_max_cut_simulator import SimulatorGraphMaxCut
from baseline.l2a_graph_max_cut_local_search import SolverLocalSearch, show_gpu_memory
from baseline.l2a_graph_utils import load_graph_list

TEN = th.Tensor


def metropolis_hastings_sampling(probs: TEN, start_xs: TEN, num_repeats: int, num_iters: int = -1) -> TEN:
    """随机平稳采样 是 metropolis-hastings sampling is:
    - 在状态转移链上 using transition kernel in Markov Chain
    - 使用随机采样估计 Monte Carlo sampling
    - 依赖接受概率去达成细致平稳条件 with accept ratio to satisfy detailed balance condition
    的采样方法。

    工程实现上:
    - 让它从多个随机的起始位置 start_xs 开始
    - 每个起始位置建立多个副本 repeat
    - 循环多次 for _ in range
    - 直到接受的样本数量超过阈值 count >= 再停止

    这种采样方法允许不同的区域能够以平稳的概率采集到符合采样概率的样本，确保样本数量比例符合期望。
    具体而言，每个区域内采集到的样本数与该区域所占的平稳分布概率成正比。
    """
    # metropolis-hastings sampling: Monte Carlo sampling using transition kernel in Markov Chain with accept ratio
    xs = start_xs.repeat(num_repeats, 1)  # 并行，，
    ps = probs.repeat(num_repeats, 1)

    num, dim = xs.shape
    device = xs.device
    num_iters = int(dim // 4) if num_iters == -1 else num_iters  # 希望至少有1/4的节点的采样结果被接受

    count = 0
    for _ in range(4):  # 迭代4轮后，即便被拒绝的节点很多，也不再迭代了。
        ids = th.randperm(dim, device=device)  # 按随机的顺序，均匀地选择节点进行采样。避免不同节点被采样的次数不同。
        for i in range(dim):
            idx = ids[i]
            chosen_p0 = ps[:, idx]
            chosen_xs = xs[:, idx]
            chosen_ps = th.where(chosen_xs, chosen_p0, 1 - chosen_p0)

            accept_rates = (1 - chosen_ps) / chosen_ps
            accept_masks = th.rand(num, device=device).lt(accept_rates)
            xs[:, idx] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

            count += accept_masks.sum()
            if count >= num * num_iters:
                break
        if count >= num * num_iters:
            break
    return xs


class McMcIterator:
    def __init__(self, num_nodes: int, num_sims: int, num_repeats: int, num_searches: int,
                 graph_list: GraphList = (), device=th.device('cpu')):
        self.num_nodes = num_nodes
        self.num_sims = num_sims
        self.num_repeats = num_repeats
        self.num_searches = num_searches
        self.device = device
        self.sim_ids = th.arange(num_sims, device=device)

        # build in reset
        self.simulator = SimulatorGraphMaxCut(graph_list=graph_list, device=self.device)  # 初始值
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)

    # 如果end to end, graph_list为空元组。如果distribution, 抽样赋值
    def reset(self, graph_list: GraphList = ()):
        self.simulator = SimulatorGraphMaxCut(graph_list=graph_list, device=self.device)
        self.searcher = SolverLocalSearch(simulator=self.simulator, num_nodes=self.num_nodes)
        self.searcher.reset(xs=self.simulator.generate_xs_randomly(num_sims=self.num_sims))

        good_xs = self.searcher.good_xs
        good_vs = self.searcher.good_vs
        return good_xs, good_vs

    # probs: 策略网络输出值
    # start_xs： 上一轮step的输出的解中，并行环境输出的最好的解
    def step(self, start_xs: TEN, probs: TEN) -> (TEN, TEN):
        xs = metropolis_hastings_sampling(probs=probs, start_xs=start_xs, num_repeats=self.num_repeats, num_iters=-1)
        vs = self.searcher.reset(xs)
        for _ in range(self.num_searches):
            # 再用local search, searcher 是local search
            xs, vs, num_update = self.searcher.random_search(num_iters=8)
        return xs, vs

    # 好的解的数量是sim数量，一个环境出一个好的解
    def pick_good_xs(self, full_xs, full_vs) -> (TEN, TEN):
        # update good_xs: use .view() instead of .reshape() for saving GPU memory
        xs_view = full_xs.view(self.num_repeats, self.num_sims, self.num_nodes)
        vs_view = full_vs.view(self.num_repeats, self.num_sims)
        ids = vs_view.argmax(dim=0)

        good_xs = xs_view[ids, self.sim_ids]
        good_vs = vs_view[ids, self.sim_ids]
        return good_xs, good_vs


'''run'''


def valid_in_single_graph(
        args0: ConfigPolicy = None,
        graph_list: GraphList = None,
        if_valid: bool = True,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    _graph_type, _num_nodes, _graph_id = 'PowerLaw', 300, 0
    args0 = args0 if args0 else ConfigPolicy(graph_type=_graph_type, num_nodes=_num_nodes)

    _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
    graph_list = load_graph_list(graph_name=_graph_name) if graph_list is None else graph_list

    '''custom'''
    # args0.num_iters = 2 ** 5 * 4
    # args0.reset_gap = 2 ** 5
    # args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
    # args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
    if os.name == 'nt':
        args0.num_sims = 2 ** 2
        args0.num_repeats = 2 ** 3

    '''config: graph'''
    graph_type = args0.graph_type
    num_nodes = args0.num_nodes

    '''config: train'''
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''model'''
    # from network import PolicyORG
    # policy_net = PolicyORG(num_nodes=num_nodes).to(device)
    from baseline.l2a_network import PolicyMLP
    policy_net = PolicyMLP(num_bits=num_nodes, mid_dim=256).to(device)
    policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net
    # from network import PolicyLSTM
    # policy_net = PolicyLSTM(inp_dim=num_nodes, mid_dim=128, out_dim=num_nodes,
    #                         embed_dim=64, num_heads=4, num_layers=4).to(device)
    # policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''iterator'''
    th.set_grad_enabled(False)
    iterator = McMcIterator(num_nodes=num_nodes, num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                            graph_list=graph_list, device=device)
    if_maximize = iterator.simulator.if_maximize

    '''evaluator'''
    save_dir = f"./ORG_{graph_type}_{num_nodes}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = iterator.reset(graph_list=graph_list)
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                          if_maximize=if_maximize)
    evaluators = []

    '''loop'''
    lamb_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float())
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = iterator.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = iterator.pick_good_xs(full_xs=full_xs, full_vs=full_vs)

        advantages = full_vs.float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)  # ↓↓↓↓↓↓↓↓↓↓ gradient
        for j in range(num_sgd_steps):
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float())
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lamb_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)  # ↑↑↑↑↑↑↑↑↑ gradient

        '''update good_xs'''
        good_i = good_vs.argmax()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} cut_value {good_vs.float().mean().long():6} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=False)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            # reset_parameters_of_model(model=policy_net)
            # good_xs, good_vs = iterator.reset(graph_list=graph_list)

            '''method2: reload (load new graph)'''
            if if_valid:
                reset_parameters_of_model(model=policy_net)
                net_params = list(policy_net.parameters())
                optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
                    else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

            _graph_id += 1
            _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
            good_xs, good_vs = iterator.reset(graph_list=load_graph_list(graph_name=_graph_name))

            evaluators.append(evaluator)
            evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                                  if_maximize=True)

    if if_valid:
        evaluator.save_record_draw_plot(fig_dpi=300)
    else:  # if_train
        th.save(policy_net.state_dict(), args0.net_path)
    return evaluators


def valid_in_single_graph_with_graph_net_and_rnn(
        args0: ConfigPolicy = None,
        args1: ConfigGraph = None,
        graph_list: GraphList = None,
        if_valid: bool = False,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    _graph_type, _num_nodes, _graph_id = 'PowerLaw', 300, 0
    args0 = args0 if args0 else ConfigPolicy(graph_type=_graph_type, num_nodes=_num_nodes)
    args1 = args1 if args1 else ConfigGraph(graph_type=_graph_type, num_nodes=_num_nodes)
    assert args0.embed_dim == args1.embed_dim
    assert args0.num_nodes == args1.num_nodes
    assert args0.graph_type == args1.graph_type

    _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
    graph_list = load_graph_list(graph_name=_graph_name) if graph_list is None else graph_list

    '''custom'''
    # args0.num_iters = 2 ** 5 * 4
    # args0.reset_gap = 2 ** 5
    # args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
    # args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
    # if os.name == 'nt':
    #     args0.num_sims = 2 ** 2
    #     args0.num_repeats = 2 ** 3

    '''config: graph'''
    graph_type = args0.graph_type
    num_nodes = args0.num_nodes

    '''config: train'''
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''model'''
    graph_net = GraphTRS(inp_dim=args1.inp_dim, mid_dim=args1.mid_dim, out_dim=args1.out_dim,
                         embed_dim=args1.embed_dim, num_heads=args1.num_heads, num_layers=args1.num_layers).to(device)
    # policy_net = args0.load_net(net_path=args0.net_path, device=device, if_valid=if_valid)
    from baseline.l2a_network import PolicyRNN
    policy_net = PolicyRNN(inp_dim=args1.inp_dim, mid_dim=args1.mid_dim, out_dim=args1.out_dim,
                           embed_dim=args1.embed_dim, num_heads=args1.num_heads, num_layers=args1.num_layers).to(device)
    policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''iterator'''
    th.set_grad_enabled(False)
    iterator = McMcIterator(num_nodes=num_nodes, num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                            graph_list=graph_list, device=device)
    if_maximize = iterator.simulator.if_maximize

    adj_bool_seq = iterator.simulator.adjacency_bool.float()[:, None, :]
    _, _, dec_node = graph_net.get_graph_information(adj_bool_seq, mask=None)
    batch_size = adj_bool_seq.shape[1]
    assert dec_node.shape == (num_nodes, batch_size, args0.embed_dim)
    del batch_size

    '''evaluator'''
    save_dir = f"./EMB_{graph_type}_{num_nodes}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = iterator.reset(graph_list=graph_list)
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                          if_maximize=if_maximize)
    evaluators = []

    '''loop'''
    lamb_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float(), dec_node=dec_node)
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = iterator.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = iterator.pick_good_xs(full_xs=full_xs, full_vs=full_vs)

        advantages = full_vs.float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)  # ↓↓↓↓↓↓↓↓↓↓ gradient
        for j in range(num_sgd_steps):
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float(), dec_node=dec_node)
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lamb_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)  # ↑↑↑↑↑↑↑↑↑ gradient

        '''update good_xs'''
        good_i = good_vs.argmax()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} cut_value {good_vs.float().mean().long():6} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            # reset_parameters_of_model(model=policy_net)
            # good_xs, good_vs = iterator.reset(graph_list=graph_list)

            '''method2: reload (load new graph)'''
            if if_valid:
                policy_net = args0.load_net(net_path=args0.net_path, device=device, if_valid=if_valid)
                policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net
                net_params = list(policy_net.parameters())
                optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
                    else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

            _graph_id += 1
            _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
            adj_bool_seq = iterator.simulator.adjacency_bool.float()[:, None, :]
            _, _, dec_node = graph_net.get_graph_information(adj_bool_seq, mask=None)
            good_xs, good_vs = iterator.reset(graph_list=load_graph_list(graph_name=_graph_name))

            evaluators.append(evaluator)
            evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                                  if_maximize=if_maximize)

    if if_valid:
        evaluator.save_record_draw_plot(fig_dpi=300)
    else:  # if_train
        th.save(policy_net.state_dict(), args0.net_path)
    return evaluators


# 单个图，或图分布，推理 训练
# 单个图，if_valid=false 搜索
# 图分布， 训练时，if_valid=false， graph_list不断更新
def valid_in_single_graph_with_graph_net_and_trs(
        args0: ConfigPolicy = None,
        args1: ConfigGraph = None,
        graph_list: GraphList = None,
        if_valid: bool = False,
):
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''dummy value'''
    _graph_type, _num_nodes, _graph_id = 'PowerLaw', 300, 0
    args0 = args0 if args0 else ConfigPolicy(graph_type=_graph_type, num_nodes=_num_nodes)
    args1 = args1 if args1 else ConfigGraph(graph_type=_graph_type, num_nodes=_num_nodes)
    assert args0.embed_dim == args1.embed_dim
    assert args0.num_nodes == args1.num_nodes
    assert args0.graph_type == args1.graph_type

    _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
    graph_list = load_graph_list(graph_name=_graph_name) if graph_list is None else graph_list

    '''custom'''
    # args0.num_iters = 2 ** 5 * 4
    # args0.reset_gap = 2 ** 5
    # args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
    # args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
    # if os.name == 'nt':
    #     args0.num_sims = 2 ** 2
    #     args0.num_repeats = 2 ** 3

    '''config: graph'''
    graph_type = args0.graph_type
    num_nodes = args0.num_nodes

    '''config: train'''
    num_sims = args0.num_sims
    num_repeats = args0.num_repeats
    num_searches = args0.num_searches
    reset_gap = args0.reset_gap
    num_iters = args0.num_iters
    num_sgd_steps = args0.num_sgd_steps
    entropy_weight = args0.entropy_weight

    weight_decay = args0.weight_decay
    learning_rate = args0.learning_rate

    show_gap = args0.show_gap

    '''model'''
    graph_net = GraphTRS(inp_dim=args1.inp_dim, mid_dim=args1.mid_dim, out_dim=args1.out_dim,
                         embed_dim=args1.embed_dim, num_heads=args1.num_heads, num_layers=args1.num_layers).to(device)
    policy_net = args0.load_net(net_path=args0.net_path, device=device, if_valid=if_valid)  # 处理embedding 向量，得到概率,

    net_params = list(policy_net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    '''iterator'''
    th.set_grad_enabled(False)
    iterator = McMcIterator(num_nodes=num_nodes, num_sims=num_sims, num_repeats=num_repeats, num_searches=num_searches,
                            graph_list=graph_list, device=device)
    if_maximize = iterator.simulator.if_maximize

    adj_bool_seq = iterator.simulator.adjacency_bool.float()[:, None, :]
    _, _, dec_node = graph_net.get_graph_information(adj_bool_seq, mask=None)
    batch_size = adj_bool_seq.shape[1]
    assert dec_node.shape == (num_nodes, batch_size, args0.embed_dim)
    del batch_size

    '''evaluator'''
    save_dir = f"./EMB_{graph_type}_{num_nodes}"
    os.makedirs(save_dir, exist_ok=True)
    good_xs, good_vs = iterator.reset(graph_list=graph_list)
    evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                          if_maximize=if_maximize)
    evaluators = []

    '''loop'''
    lambda_entropy = (th.cos(th.arange(reset_gap, device=device) / (reset_gap - 1) * th.pi) + 1) / 2 * entropy_weight
    for i in range(num_iters):
        # probs
        probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float(), dec_node=dec_node)
        probs = probs.repeat(num_sims, 1)

        full_xs, full_vs = iterator.step(start_xs=good_xs, probs=probs)
        good_xs, good_vs = iterator.pick_good_xs(full_xs=full_xs, full_vs=full_vs)

        advantages = full_vs.float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        del full_vs

        th.set_grad_enabled(True)  # ↓↓↓↓↓↓↓↓↓↓ gradient
        for j in range(num_sgd_steps):  # 提高样本利用效率，可以不循环
            probs = policy_net.auto_regressive(xs_flt=good_xs[good_vs.argmax(), None, :].float(), dec_node=dec_node)
            probs = probs.repeat(num_sims, 1)

            full_ps = probs.repeat(num_repeats, 1)
            logprobs = th.log(th.where(full_xs, full_ps, 1 - full_ps)).sum(dim=1)

            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)  # entropy有用，更稳定
            obj_entropy = entropy.mean()
            obj_values = (th.softmax(logprobs, dim=0) * advantages).sum()

            objective = obj_values + obj_entropy * lambda_entropy[i % reset_gap]
            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()
        th.set_grad_enabled(False)  # ↑↑↑↑↑↑↑↑↑ gradient

        '''update good_xs'''
        good_i = good_vs.argmax()
        good_x = good_xs[good_i]
        good_v = good_vs[good_i]
        if_show_x = evaluator.record2(i=i, vs=good_v.item(), xs=good_x)

        if (i + 1) % show_gap == 0 or if_show_x:
            _probs = 1 - probs
            entropy = (probs * probs.log2() + _probs * _probs.log2()).mean(dim=1)
            obj_entropy = -entropy.mean().item()

            show_str = f"| entropy {obj_entropy:9.4f} cut_value {good_vs.float().mean().long():6} < {good_vs.max():6}"
            evaluator.logging_print(show_str=show_str, if_show_x=if_show_x)
            sys.stdout.flush()

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)} "
                  f"| up_rate {evaluator.best_v / evaluator.first_v - 1.:8.5f}")
            sys.stdout.flush()

            '''method1: reset (keep old graph)'''
            # reset_parameters_of_model(model=policy_net)
            # good_xs, good_vs = iterator.reset(graph_list=graph_list)

            '''method2: reload (load new graph)'''
            if if_valid:
                policy_net = args0.load_net(net_path=args0.net_path, device=device, if_valid=if_valid)
                policy_net = th.compile(policy_net) if th.__version__ < '2.0' else policy_net
                net_params = list(policy_net.parameters())
                optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
                    else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

            _graph_id += 1
            _graph_name = f'{args0.graph_type}_{args0.num_nodes}_ID{_graph_id}'
            adj_bool_seq = iterator.simulator.adjacency_bool.float()[:, None, :]
            _, _, dec_node = graph_net.get_graph_information(adj_bool_seq, mask=None)
            good_xs, good_vs = iterator.reset(graph_list=load_graph_list(graph_name=_graph_name))

            evaluators.append(evaluator)
            evaluator = Evaluator(save_dir=save_dir, num_bits=num_nodes, x=good_xs[0], v=good_vs[0].item(),
                                  if_maximize=if_maximize)

    if if_valid:
        evaluator.save_record_draw_plot(fig_dpi=300)
    else:  # if_train
        th.save(policy_net.state_dict(), args0.net_path)
    return evaluators


def run_mlp(graph_type='ErdosRenyi'):
    assert graph_type in ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']

    logging_path = f'L2A_MLP_{graph_type}.log'  # todo different from run_trs()
    os.remove(logging_path) if os.path.exists(logging_path) else None
    logging.basicConfig(filename=logging_path, level=logging.INFO)

    model_dir = './model/mlp'  # todo different from run_trs()
    os.makedirs(model_dir, exist_ok=True)

    for num_nodes in (600, 500, 400, 300, 200, 100):
        # args1 = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)  # todo different from run_trs()
        args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
        args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"
        args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
        args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
        if num_nodes >= 500:
            args0.num_iters = 2 ** 6 * 30
            args0.reset_gap = 2 ** 6
            per_second = 30
        else:
            args0.num_iters = 2 ** 5 * 30
            args0.reset_gap = 2 ** 5
            per_second = 10

        evaluators = valid_in_single_graph(args0=args0, if_valid=True)  # todo different from run_trs()

        for graph_id, evaluator in enumerate(evaluators):
            x_str = evaluator.encoder_base64.bool_to_str(evaluator.best_x).replace('\n', '')
            valid_str = read_info_from_recorder(evaluator.recorder2, per_second=per_second)

            info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
            print(f"|INFO: {info_str}")
            sys.stdout.flush()

            logging.info(info_str)

    logging.shutdown()


def run_rnn(graph_type='ErdosRenyi'):
    assert graph_type in ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']

    logging_path = f'L2A_TRS_{graph_type}.log'
    os.remove(logging_path) if os.path.exists(logging_path) else None
    logging.basicConfig(filename=logging_path, level=logging.INFO)

    model_dir = './model/rnn'  # todo
    os.makedirs(model_dir, exist_ok=True)

    for num_nodes in (600, 500, 400, 300, 200, 100):
        args1 = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)
        args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
        args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"
        args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
        args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
        if num_nodes >= 500:
            args0.num_iters = 2 ** 6 * 30
            args0.reset_gap = 2 ** 6
            per_second = 30
        else:
            args0.num_iters = 2 ** 5 * 30
            args0.reset_gap = 2 ** 5
            per_second = 10

        evaluators = valid_in_single_graph_with_graph_net_and_rnn(args0=args0, args1=args1, if_valid=True)

        for graph_id, evaluator in enumerate(evaluators):
            x_str = evaluator.encoder_base64.bool_to_str(evaluator.best_x).replace('\n', '')
            valid_str = read_info_from_recorder(evaluator.recorder2, per_second=per_second)

            info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
            print(f"|INFO: {info_str}")
            sys.stdout.flush()

            logging.info(info_str)

    logging.shutdown()


def run_trs(graph_type='ErdosRenyi'):
    assert graph_type in ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']

    logging_path = f'L2A_TRS_{graph_type}.log'
    os.remove(logging_path) if os.path.exists(logging_path) else None
    logging.basicConfig(filename=logging_path, level=logging.INFO)

    model_dir = './model/trs'  # todo
    os.makedirs(model_dir, exist_ok=True)

    for num_nodes in (600, 500, 400, 300, 200, 100):
        args1 = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)
        args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
        args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"
        args0.num_sims = 2 ** 6  # LocalSearch 的初始解数量
        args0.num_repeats = 2 ** 6  # LocalSearch 对于每以个初始解进行复制的数量
        if num_nodes >= 500:
            args0.num_iters = 2 ** 6 * 30
            args0.reset_gap = 2 ** 6
            per_second = 30
        else:
            args0.num_iters = 2 ** 5 * 30
            args0.reset_gap = 2 ** 5
            per_second = 10

        evaluators = valid_in_single_graph_with_graph_net_and_trs(args0=args0, args1=args1, if_valid=True)

        for graph_id, evaluator in enumerate(evaluators):
            x_str = evaluator.encoder_base64.bool_to_str(evaluator.best_x).replace('\n', '')
            valid_str = read_info_from_recorder(evaluator.recorder2, per_second=per_second)

            info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
            print(f"|INFO: {info_str}")
            sys.stdout.flush()

            logging.info(info_str)

    logging.shutdown()


def end_to_end_demo_input_graph_list_then_output_x_using_mlp():
    graph_type = 'PowerLaw'  # ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']
    num_nodes = 300
    model_dir = './model/mlp'  # ['model/mlp', 'model/lstm', 'model/trs']
    graph_id = 0

    '''input'''
    graph_list: GraphList = load_graph_list(graph_name=f"{graph_type}_{num_nodes}_ID{graph_id}")

    """MLP"""
    args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
    args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"

    evaluator = valid_in_single_graph(args0=args0, graph_list=graph_list, if_valid=True)[0]

    '''MLP output'''
    x = evaluator.best_x
    x_str = evaluator.best_x_str()
    valid_str = read_info_from_recorder(evaluator.recorder2, per_second=10)

    info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
    print(f"|INFO: {info_str}")
    print(f"|Solution X {x}")


def end_to_end_demo_input_graph_list_then_output_x_using_rnn():
    graph_type = 'PowerLaw'  # ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']
    num_nodes = 300
    model_dir = './model/mlp'  # ['model/mlp', 'model/lstm', 'model/trs']
    graph_id = 0

    '''input'''
    graph_list: GraphList = load_graph_list(graph_name=f"{graph_type}_{num_nodes}_ID{graph_id}")

    """MLP"""
    args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
    args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"

    evaluator = valid_in_single_graph_with_graph_net_and_rnn(args0=args0, graph_list=graph_list, if_valid=True)[0]

    '''MLP output'''
    x = evaluator.best_x
    x_str = evaluator.best_x_str()
    valid_str = read_info_from_recorder(evaluator.recorder2, per_second=10)

    info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
    print(f"|INFO: {info_str}")
    print(f"|Solution X {x}")


def end_to_end_demo_input_graph_list_then_output_x_using_trs():
    graph_type = 'PowerLaw'  # ['ErdosRenyi', 'BarabasiAlbert', 'PowerLaw']
    num_nodes = 300
    model_dir = './model/trs'  # ['model/mlp', 'model/lstm', 'model/trs']
    graph_id = 0

    '''input'''
    graph_list: GraphList = load_graph_list(graph_name=f"{graph_type}_{num_nodes}_ID{graph_id}")

    """TRS"""
    args1 = ConfigGraph(graph_type=graph_type, num_nodes=num_nodes)
    args0 = ConfigPolicy(graph_type=graph_type, num_nodes=num_nodes)
    args0.net_path = f"{model_dir}/policy_net_{graph_type}_Node{num_nodes}.pth"
    assert args0.reset_gap == args0.num_iters  # to keep old graph

    # 结果返回到evaluator，打印出来
    evaluator = valid_in_single_graph_with_graph_net_and_trs(args0=args0, args1=args1,
                                                             graph_list=graph_list, if_valid=True)[0]

    '''TRS output'''
    x = evaluator.best_x
    x_str = evaluator.best_x_str()
    valid_str = read_info_from_recorder(evaluator.recorder2, per_second=10)

    info_str = f"{graph_type} {num_nodes} {graph_id} {x_str} {valid_str}"
    print(f"|INFO: {info_str}")
    print(f"|Solution X {x}")


if __name__ == '__main__':
    end_to_end_demo_input_graph_list_then_output_x_using_mlp()  # an end-to-end tutorial
    end_to_end_demo_input_graph_list_then_output_x_using_rnn()  # an end-to-end tutorial
    end_to_end_demo_input_graph_list_then_output_x_using_trs()  # an end-to-end tutorial

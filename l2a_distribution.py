import os
import sys
import time
import math
import json
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm
from evaluator import Evaluator0
from simulator import MaxcutSimulator
from util import (load_graph_from_txt,
                    save_graph_info_to_txt,
                    generate_graph,
                    generate_graph_for_validation,
                    load_graph,
                    EncoderBase64,
                    calc_device
                  )
from config import *

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from net import OptimizerLSTM, OptimizerLSTM2

'''Variable Naming Conventions
edge: An edge within the graph.
Each edge is directed from node0 to node1. In an undirected graph, the endpoints can be arbitrarily fixed in order.
n0: Index of the left endpoint node0.
n1: Index of the right endpoint node1.
dt: Distance which is the length of the edge, that is, the distance between node0 and node1.

The Max Cut problem divides nodes into two sets, set0 and set1.
p0: The probability that endpoint node0 belongs to set0, (1-p0): The probability that endpoint node0 belongs to set1.
p1: The probability that endpoint node1 belongs to set0, (1-p1): The probability that endpoint node1 belongs to set1.

prob: Probability that marks the probability of each node in the order of node0. It is the probabilistic form of the solution to the GraphMaxCut problem.
sln_x: solution_x that marks the set to which each node belongs in the order of node0. It is the binary representation of the solution to the GraphMaxCut problem.
'''

TEN = th.Tensor
INT = th.IntTensor


class Agent:  # Demo
    def __init__(self, graph_name='powerlaw_64', gpu_id: int = 0, json_path: str = '', graph_tuple=None):
        pass

        """数据超参数（路径，存储，预处理）"""
        self.json_path = f'./GraphMaxCut_{graph_name}.json'  # 存放超参数的json文件。将会保存所有类似为 str int float 的参数
        self.graph_name = graph_name
        self.gpu_id = gpu_id

        '''GPU memory'''
        self.num_envs = 2 ** 6
        self.mid_dim = 2 ** 6
        self.num_layers = 2
        self.seq_len = 2 ** 6
        self.reset_gap = 2 ** 6
        self.learning_rate = 1e-3

        '''train and evaluate'''
        self.num_opti = 2 ** 16
        self.eval_gap = 2 ** 2
        self.save_dir = f"./result_{graph_name}_{gpu_id}"

        if graph_tuple is None:
            if os.path.exists(json_path):
                self.load_from_json(json_path=json_path)
                self.save_as_json(json_path=json_path)
            vars_str = str(vars(self)).replace(", '", ", \n'")
            print(f"| Config\n{vars_str}")

        """agent"""
        '''init simulator'''
        self.sim = MaxcutSimulator(graph_name=graph_name, gpu_id=gpu_id, graph_tuple=graph_tuple)
        self.enc = EncoderBase64(num_nodes=self.sim.num_nodes)
        self.num_nodes = self.sim.num_nodes
        self.num_edges = self.sim.num_edges

        '''init optimizer'''
        self.device = calc_device(gpu_id)
        self.opt_opti = OptimizerLSTM(
            inp_dim=self.num_nodes,
            mid_dim=self.mid_dim,
            out_dim=self.num_nodes,
            num_layers=self.num_layers
        ).to(self.device)
        self.opt_base = th.optim.Adam(self.opt_opti.parameters(), lr=self.learning_rate)

        '''init evaluator'''
        self.evaluator = Evaluator0(sim=self.sim, enc=self.enc)

    def iter_reset(self):
        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None
        return probs, hidden0, hidden1

    def iter_loop(self, probs, hidden0, hidden1):
        prob_ = probs.clone()
        updates = []

        for j in range(self.seq_len):
            obj = self.sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = self.opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * self.learning_rate
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        return prob_, hidden0, hidden1

    def search(self, j):
        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None

        total_i = j * self.reset_gap
        for i in range(self.reset_gap):
            prob_, hidden0, hidden1 = self.iter_loop(probs=probs, hidden0=hidden0, hidden1=hidden1)

            obj_ = self.sim.get_objectives(prob_).mean()
            self.opt_base.zero_grad()
            obj_.backward()
            self.opt_base.step()

            probs.data[:] = prob_

            if i % self.eval_gap:
                sln_xs = self.sim.prob_to_bool(probs)
                self.evaluator.evaluate_and_print(solutions=sln_xs, i=total_i + i, obj=obj_)

        best_solution = self.evaluator.best_solution
        best_score = self.evaluator.best_score
        return best_solution, best_score

    def load_from_json(self, json_path: str):
        with open(json_path, "r") as file:
            json_dict = json.load(file)
        for key, value in json_dict.items():
            setattr(self, key, value)

    def save_as_json(self, json_path: str):
        json_dict = dict([(key, value) for key, value in vars(self).items()
                          if type(value) in {str, int, float}])
        print(json_dict)
        with open(json_path, "w") as file:
            json.dump(json_dict, file, indent=4)


class AgentDist(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_opti = OptimizerLSTM2(
            inp_dim=(self.num_nodes, self.sim.adjacency_vector.shape[0]),
            mid_dim=self.mid_dim,
            out_dim=self.num_nodes,
            num_layers=self.num_layers
        ).to(self.device)
        self.opt_base = th.optim.Adam(self.opt_opti.parameters(), lr=self.learning_rate)

    def iter_loop(self, probs, hidden0, hidden1):
        prob_ = probs.clone()
        updates = []
        graph_tensor = self.sim.adjacency_vector

        for j in range(self.seq_len):
            obj = self.sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = self.opt_opti(
                grad_s.unsqueeze(0),
                graph_tensor.unsqueeze(0),
                hidden0,
                hidden1,
            )  # todo graph dist
            update = (update.squeeze_(0) - grad_s) * self.learning_rate
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        return prob_, hidden0, hidden1

    def search_for_valid(self, j):
        self.opt_base = None

        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None

        total_i = j * self.reset_gap
        for i in range(self.reset_gap):
            prob_, hidden0, hidden1 = self.iter_loop(probs=probs, hidden0=hidden0, hidden1=hidden1)
            probs.data[:] = prob_

            if i % self.eval_gap:
                solutions = self.sim.prob_to_bool(probs)
                self.evaluator.evaluate_and_print(solutions=solutions, i=total_i + i, obj=th.tensor(0))

        best_solution = self.evaluator.best_solution
        best_score = self.evaluator.best_score
        return best_solution, best_score

def run():
    gpu_id = GPU_ID
    num_nodes = 50
    graph_name = f"powerlaw_{num_nodes}"

    # todo graph dist
    num_valid, seed_num = 100, 0
    np.random.seed(seed_num)
    th.manual_seed(seed_num)
    g_type, num_nodes = graph_name.split('_')
    num_nodes = int(num_nodes)
    graphs = []
    agents = []
    for i in range(num_valid):
        graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
        agent = AgentDist(graph_name=graph_name, gpu_id=gpu_id, json_path='auto_build',
                          graph_tuple=(graph, num_nodes, num_edges))

        graphs.append(graph)
        agents.append(agent)
    valid_opt_opti = agents[0].opt_opti
    for i in range(1, num_valid):
        agent = agents[i]
        agent.opt_opti = valid_opt_opti
        agent.opt_base = None

    agent = AgentDist(graph_name=graph_name, gpu_id=gpu_id, json_path='auto_build')
    valid_opt_opti.load_state_dict(agent.opt_opti.state_dict())

    best_solutions: list = [0, ] * num_valid
    best_scores: list = [0, ] * num_valid
    for j in range(agent.num_opti):
        agent.sim.__init__(graph_name=graph_name, gpu_id=gpu_id)
        _best_solution, _best_score = agent.search(j=j)

        valid_opt_opti.load_state_dict(agent.opt_opti.state_dict())
        for j_valid in range(num_valid):
            _agent = agents[j_valid]
            best_solution, best_score = _agent.search_for_valid(j=j)
            best_solutions[j_valid] = best_solution
            best_scores[j_valid] = best_score
        print(f"| best_scores.avg {sum(best_scores) / len(best_scores)}")

    print(f"\nbest_solution {best_solutions}"
          f"\nbest_score {best_scores}")


if __name__ == '__main__':
    run()

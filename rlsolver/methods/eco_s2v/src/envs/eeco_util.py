import random
from abc import ABC, abstractmethod
from enum import Enum

import networkx as nx
import numpy as np
import torch
from rlsolver.methods.eco_s2v.config.config import *

class EdgeType(Enum):

    UNIFORM = 1
    DISCRETE = 2
    RANDOM = 3

class RewardSignal(Enum):

    DENSE = 1
    BLS = 2
    SINGLE = 3
    CUSTOM_BLS = 4

class ExtraAction(Enum):

    PASS = 1
    RANDOMISE = 2
    NONE = 3

class OptimisationTarget(Enum):

    CUT = 1
    ENERGY = 2

class SpinBasis(Enum):

    SIGNED = 1
    BINARY = 2

class Observable(Enum):
    # Local observations that differ between nodes.
    SPIN_STATE = 1
    IMMEDIATE_REWARD_AVAILABLE = 2
    TIME_SINCE_FLIP = 3

    # Global observations that are the same for all nodes.
    EPISODE_TIME = 4
    TERMINATION_IMMANENCY = 5
    NUMBER_OF_GREEDY_ACTIONS_AVAILABLE = 6
    DISTANCE_FROM_BEST_SCORE = 7
    DISTANCE_FROM_BEST_STATE = 8

DEFAULT_OBSERVABLES = [Observable.SPIN_STATE,
                       Observable.IMMEDIATE_REWARD_AVAILABLE,
                       Observable.TIME_SINCE_FLIP,
                       Observable.DISTANCE_FROM_BEST_SCORE,
                       Observable.DISTANCE_FROM_BEST_STATE,
                       Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE,
                       Observable.TERMINATION_IMMANENCY]

class GraphGenerator(ABC):

    def __init__(self, n_spins, edge_type, biased=False,n_graphs=2**3):
        self.n_spins = n_spins
        self.edge_type = edge_type
        self.biased = biased
        self.n_graphs = n_graphs

    def pad_matrix(self, matrix):
        dim = matrix.shape[0]
        m = np.zeros((dim+1,dim+1))
        m[:-1,:-1] = matrix
        return matrix

    def pad_bias(self, bias):
        return np.concatenate((bias,[0]))

    @abstractmethod
    def get(self, with_padding=False):
        raise NotImplementedError

###################
# Unbiased graphs #
###################
class RandomGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if self.edge_type == EdgeType.UNIFORM:
            self.get_w = lambda : 1
        elif self.edge_type == EdgeType.DISCRETE:
            self.get_w = lambda : np.random.choice([+1, -1])
        elif self.edge_type == EdgeType.RANDOM:
            self.get_w = lambda : np.random.uniform(-1, 1)
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):

        g_size = self.n_spins

        density = np.random.uniform()
        matrix = np.zeros((g_size, g_size))
        for i in range(self.n_spins):
            for j in range(i):
                if np.random.uniform() < density:
                    w = self.get_w()
                    matrix[i, j] = w
                    matrix[j, i] = w

        matrix = self.pad_matrix(matrix) if with_padding else matrix

        if self.biased:
            bias = np.array([self.get_w() if np.random.uniform() < density else 0 for _ in range(self.n_spins)])
            bias = self.pad_bias(bias) if with_padding else bias
            return matrix, bias
        else:
            return matrix

        m = self.pad_matrix(self.matrix) if with_padding else self.matrix

        if self.biased:
            b = self.pad_bias(self.bias) if with_padding else self.bias
            return m, b
        else:
            return m

class RandomErdosRenyiGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, p_connection=[0.1,0], edge_type=EdgeType.DISCRETE):
        super().__init__(n_spins, edge_type, False)

        if type(p_connection) not in [list,tuple]:
            p_connection = [p_connection, 0]
        assert len(p_connection)==2, "p_connection must have length 2"
        self.p_connection  = p_connection

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda : np.ones((self.n_spins,self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2.*np.random.rand(self.n_spins,self.n_spins)-1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):

        p = np.clip(np.random.normal(*self.p_connection),0,1)

        g = nx.erdos_renyi_graph(self.n_spins, p)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        # No self-connections (this modifies adj in-place).
        np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj

class RandomBarabasiAlbertGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, m_insertion_edges=4, edge_type=EdgeType.DISCRETE, n_graphs=2**3):
        super().__init__(n_spins, edge_type, False,n_graphs)

        self.m_insertion_edges = m_insertion_edges
        self.device = TRAIN_DEVICE
        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: torch.ones((self.n_spins, self.n_spins), device=self.device)
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_spins, self.n_spins), device=self.device) - 1.
                mask = torch.tril(mask) + torch.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * torch.randint(0, 2, (self.n_graphs, n_spins, n_spins), dtype=torch.float32,device=self.device) - 1
                mask = torch.tril(mask, diagonal=0) + torch.triu(mask.transpose(1, 2), diagonal=1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        adj = torch.empty((self.n_graphs,self.n_spins,self.n_spins),device=self.device,dtype=torch.float)
        for i in range(self.n_graphs):
            g = nx.barabasi_albert_graph(self.n_spins, self.m_insertion_edges)
            adj[i] = torch.tensor(nx.to_numpy_array(g), dtype=torch.float32,device=self.device).fill_diagonal_(0)

        adj = adj*self.get_connection_mask()
        # No self-connections (this modifies adj in-place).
        return self.pad_matrix(adj) if with_padding else adj

class RandomRegularGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, d_node=[2,0], edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if type(d_node) not in [list,tuple]:
            d_node = [d_node, 0]
        assert len(d_node)==2, "k_neighbours must have length 2"
        self.d_node  = d_node

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda : np.ones((self.n_spins,self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2.*np.random.rand(self.n_spins,self.n_spins)-1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        k = np.clip(int(np.random.normal(*self.d_node)),0,self.n_spins)

        g = nx.random_regular_graph(k, self.n_spins)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        if not self.biased:
            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj

class RandomWattsStrogatzGraphGenerator(GraphGenerator):

    def __init__(self, n_spins=20, k_neighbours=[2,0], edge_type=EdgeType.DISCRETE, biased=False):
        super().__init__(n_spins, edge_type, biased)

        if type(k_neighbours) not in [list,tuple]:
            k_neighbours = [k_neighbours, 0]
        assert len(k_neighbours)==2, "k_neighbours must have length 2"
        self.k_neighbours  = k_neighbours

        if self.edge_type == EdgeType.UNIFORM:
            self.get_connection_mask = lambda: np.ones((self.n_spins, self.n_spins))
        elif self.edge_type == EdgeType.DISCRETE:
            def get_connection_mask():
                mask = 2. * np.random.randint(2, size=(self.n_spins, self.n_spins)) - 1.
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        elif self.edge_type == EdgeType.RANDOM:
            def get_connection_mask():
                mask = 2. * np.random.rand(self.n_spins, self.n_spins) - 1
                mask = np.tril(mask) + np.triu(mask.T, 1)
                return mask
            self.get_connection_mask = get_connection_mask
        else:
            raise NotImplementedError()

    def get(self, with_padding=False):
        k = np.clip(int(np.random.normal(*self.k_neighbours)),0,self.n_spins)

        g = nx.watts_strogatz_graph(self.n_spins, k, 0)
        adj = np.multiply(nx.to_numpy_array(g), self.get_connection_mask())

        if not self.biased:
            # No self-connections (this modifies adj in-place).
            np.fill_diagonal(adj, 0)

        return self.pad_matrix(adj) if with_padding else adj

################
# Known graphs #
################
class SingleGraphGenerator(GraphGenerator):

    def __init__(self, matrix, bias=None):

        n_spins = matrix.shape[0]

        if  np.isin(matrix,[0,1]).all():
            edge_type=EdgeType.UNIFORM
        elif  np.isin(matrix,[0,-1,1]).all():
            edge_type=EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, bias is not None)

        self.matrix = matrix
        self.bias = bias

    def get(self, with_padding=False):

        m = self.pad_matrix(self.matrix) if with_padding else self.matrix

        if self.biased:
            b = self.pad_bias(self.bias) if with_padding else self.bias
            return m, b
        else:
            return m

class SetGraphGenerator(GraphGenerator):

    def __init__(self, matrices, biases=None, ordered=False):

        if len(set([m.shape[0]-1 for m in matrices]))==1:
            n_spins = matrices[0].shape[0]
        else:
            raise NotImplementedError("All graphs in SetGraphGenerator must have the same dimension.")

        if all([torch.isin(m, torch.tensor([0, 1],device=TRAIN_DEVICE)).all().item() for m in matrices]):

            edge_type=EdgeType.UNIFORM
        elif all([torch.isin(m, torch.tensor([0,-1, 1],device=TRAIN_DEVICE)).all().item() for m in matrices]):
            edge_type=EdgeType.DISCRETE
        else:
            edge_type = EdgeType.RANDOM

        super().__init__(n_spins, edge_type, biases is not None)

        if not self.biased:
            self.graphs = matrices
        else:
            assert len(matrices)==len(biases), "Must pass through the same number of matrices and biases."
            assert all([len(b)==self.n_spins+1 for b in biases]), "All biases and must have the same dimension as the matrices."
            self.graphs = list(zip(matrices, biases))

        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self, with_padding=False):
        m = self.graphs
        return self.pad_matrix(m) if with_padding else m


class PerturbedGraphGenerator(GraphGenerator):

    def __init__(self, matrices, perturb_mean=0, perturb_std=0.01, biases=None, ordered=False):

        if type(matrices) != list:
            matrices = list(matrices)

        if biases is not None:
            if type(biases) != list:
                biases = list(biases)

        if len(set([m.shape[0] - 1 for m in matrices])) == 1:
            n_spins = matrices[0].shape[0]
        else:
            raise NotImplementedError("All graphs passed to PerturbedGraphGenerator must have the same dimension.")

        super().__init__(n_spins, EdgeType.RANDOM, biases is not None)

        self.perturb_mean = perturb_mean
        self.perturb_std = perturb_std

        if not self.biased:
            self.graphs = matrices
        else:
            raise NotImplementedError("Not implemented PerturbedGraphGenerator for biased graphs yet.")

        self.ordered = ordered
        if self.ordered:
            self.i = 0

    def get(self, with_padding=False):
        if self.ordered:
            m = self.graphs[self.i]
            self.i = (self.i + 1)%len(self.graphs)
            if self.biased:
                m, b = m
        else:
            if not self.biased:
                m = random.sample(self.graphs, k=1)[0]
            else:
                m, b = random.sample(self.graphs, k=1)[0]

        # Sample noise.
        noise = np.random.normal(self.perturb_mean, self.perturb_std, size=m.shape)
        # Set noise to 0 for non-edges in the adjacency matrix.
        np.putmask(noise, m == 0, 0)
        # Ensure noise is symettric.
        noise = np.tril(noise) + np.triu(noise.T, 1)

        m = m + noise

        return self.pad_matrix(m) if with_padding else m

# class HistoryBuffer():
#     def __init__(self):
#         self.buffer = {}
#         self.current_action_hist = set([])
#         self.current_action_hist_len = 0
#
#     def update(self, action):
#         new_action_hist = self.current_action_hist.copy()
#         if action in self.current_action_hist:
#             new_action_hist.remove(action)
#             self.current_action_hist_len -= 1
#         else:
#             new_action_hist.add(action)
#             self.current_action_hist_len += 1
#
#         try:
#             list_of_states = self.buffer[self.current_action_hist_len]
#             if new_action_hist in list_of_states:
#                 self.current_action_hist = new_action_hist
#                 return False
#         except KeyError:
#             list_of_states = []
#
#         list_of_states.append(new_action_hist)
#         self.current_action_hist = new_action_hist
#         self.buffer[self.current_action_hist_len] = list_of_states
#         return True


class HistoryBuffer:
    def __init__(self,n_sims,n_graphs):
        """
        env_shape: (N1, N2) 表示有 N1 x N2 个环境。

        self.buffers[i][j]: 一个字典，用来存储不同 "动作集合大小" -> "已出现过的 frozenset 集合"。
        self.current_action_hists[i][j]: 当前 (i, j) 环境已收集的动作集合 (set)。
        self.current_action_hist_lens[i][j]: 上述动作集合大小 (int)。
        """
        self.env_shape = n_graphs,n_sims
        N1, N2 = self.env_shape

        # buffers[i][j] = { hist_len: set_of_frozensets }
        self.buffers = [
            [dict() for _ in range(N2)]
            for _ in range(N1)
        ]

        # 当前每个环境的动作集合与长度
        self.current_action_hists = [
            [set() for _ in range(N2)]
            for _ in range(N1)
        ]
        self.current_action_hist_lens = [
            [0 for _ in range(N2)]
            for _ in range(N1)
        ]

    def update(self, actions: torch.Tensor):
        """
        actions: 形状 (N1, N2)
          - actions[i, j] 是第 (i, j) 个环境在本次 step 的动作(标量或单元素)。

        返回：一个与 actions 同形状的布尔张量 (N1, N2)，
             表示该步更新后，每个环境是否出现了“新的动作集合状态”（True 表示新出现）。
        """
        # 检查传入的 actions 形状是否与 env_shape 对应
        N1, N2 = self.env_shape
        assert actions.shape == (N1, N2), (
            f"actions 的形状 {actions.shape} 不符合 env_shape {self.env_shape}"
        )

        # 用于存储所有环境的返回结果 (True/False)
        # 这里用 torch.bool 张量来保存
        new_state_flags = torch.zeros((N1, N2), dtype=torch.bool)
        actions_list = actions.tolist()
        # 遍历所有二维环境
        for i in range(N1):
            for j in range(N2):
                # 取出第 (i, j) 个环境的动作
                # 若是标量，actions[i,j] 可能是一个 0-dim tensor，因此 .item() 取出 Python 数值
                env_action = actions_list[i][j]

                # 拷贝旧的动作集合
                new_action_hist = self.current_action_hists[i][j].copy()
                env_action_hist_len = self.current_action_hist_lens[i][j]

                # 如果这个动作已在集合中，则移除，否则添加
                if env_action in new_action_hist:
                    new_action_hist.remove(env_action)
                    env_action_hist_len -= 1
                else:
                    new_action_hist.add(env_action)
                    env_action_hist_len += 1

                # 转成 frozenset 用于可哈希存储
                f_new_action_hist = frozenset(new_action_hist)

                # 从 buffer 中检查该大小下是否已出现过这个 frozenset
                try:
                    set_of_states = self.buffers[i][j][env_action_hist_len]
                    if f_new_action_hist in set_of_states:
                        # 已出现过，new_state_flags[i, j] = False
                        # 只要更新 current_action_hist / lens 就行
                        self.current_action_hists[i][j] = new_action_hist
                        self.current_action_hist_lens[i][j] = env_action_hist_len
                        continue  # 跳过后续逻辑
                except KeyError:
                    # 如果还没有对应大小的记录，则初始化
                    set_of_states = set()

                # 如果能走到这里，说明是新的动作集合状态
                set_of_states.add(f_new_action_hist)
                self.buffers[i][j][env_action_hist_len] = set_of_states

                # 更新当前动作集合
                self.current_action_hists[i][j] = new_action_hist
                self.current_action_hist_lens[i][j] = env_action_hist_len

                # 标记 True
                new_state_flags[i, j] = True

        return new_state_flags

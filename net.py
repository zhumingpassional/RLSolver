import os
import sys
import time
import math
import json
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn


try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class Net(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(num_nodes, embedding_dim)
        nn.init.orthogonal_(self.embedding_layer.weight)

    def forward(self, ids):
        embedded_x = self.embedding_layer(ids)
        return embedded_x

class PolicyMLP(nn.Module):
    def __init__(self, node_dim, mid_dim, embedding_dim):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(node_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim), )
        self.net2 = nn.Sequential(nn.Linear(mid_dim + embedding_dim, mid_dim), nn.GELU(), nn.LayerNorm(mid_dim),
                                  nn.Linear(mid_dim, 1), nn.Sigmoid(), )

    def forward(self, xs, embedding_ws):
        xs1 = self.net1(xs)
        xs2 = self.net2(th.concat((xs1, embedding_ws), dim=1))
        return xs2.squeeze(1)


class OptimizerLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


class OptimizerLSTM2(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        inp_dim0 = inp_dim[0]
        inp_dim1 = inp_dim[1]

        self.rnn0 = nn.LSTM(inp_dim0, mid_dim, num_layers=num_layers)
        self.enc0 = nn.Sequential(nn.Linear(inp_dim1, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, mid_dim))  # todo graph dist
        self.mlp0 = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, out_dim))

        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, graph, hid0=None, hid1=None):
        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        tmp0, hid0 = self.rnn0(inp, hid0)

        graph = self.enc0(graph) * th.ones((d0, d1, 1), device=tmp0.device)
        tmp0 = th.concat((tmp0, graph), dim=2)  # todo graph dist
        out0 = self.mlp0(tmp0)

        out = out0 + out1
        return out, hid0, hid1

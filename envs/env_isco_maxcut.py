import torch
import pickle
import torch
import torch.nn.functional as F
import math
import networkx as nx
import os


class Maxcut:
    def __init__(self):
        self.init_temperature = 1.0
        self.chain_length = 50000
        self.batch_size = 32
        self.max_num_nodes =2000
        self.max_num_edges = 19990
        self.num_instances = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_data_list(self,filename):
        with open(filename, 'r') as file:
            lines = []
            line = file.readline()  # 读取第一行
            while line is not None and line != '':
                if '//' not in line:
                    lines.append(line)
                line = file.readline()  # 读取下一行
            # lines = file.readlines()
            lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
        num_nodes, num_edges = lines[0]
        g = nx.Graph()
        nodes = list(range(num_nodes))
        g.add_nodes_from(nodes)
        for item in lines[1:]:
            g.add_edge(item[0] - 1, item[1] - 1, weight=item[2])
        edge_from = torch.zeros((self.num_instances,num_edges,), dtype=torch.int32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        edge_to = torch.zeros((self.num_instances,num_edges,), dtype=torch.int32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        edge_weight = torch.zeros((self.num_instances,num_edges,), dtype=torch.int32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i, e in enumerate(g.edges(data=True)):
            x, y = e[0], e[1]
            edge_from[0][i] = x
            edge_to[0][i] = y
            edge_weight[0][i] = e[2]['weight']

        tensor_dict = {
                            'edge_from': edge_from,
                            'edge_to': edge_to,
                            'edge_weight': edge_weight,
                        }
        data_list = []
        return data_list,tensor_dict
    

    def get_energy(self,tensor_dict,sample):
        edge_from = tensor_dict["edge_from"].long()
        edge_to = tensor_dict["edge_to"].long()
        edge_weight = tensor_dict["edge_weight"].float()
        x = sample.clone().detach().requires_grad_(True)
        delta_x  = (x * 2 - 1)
        gather2src = torch.gather(delta_x, 1,  edge_from)
        gather2dst = torch.gather(delta_x, 1, edge_to)
        is_cut = (1 - gather2src * gather2dst) / 2.0
        energy = torch.sum(is_cut * edge_weight,dim=-1)
        energy = - energy
        grad = torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=torch.ones_like(energy), create_graph=False)
        grad = grad[0]

        return energy,grad

    


def build_model():
    return Maxcut()
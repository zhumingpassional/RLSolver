import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNN_A2C(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[]):
        super().__init__()

        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights

        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )

        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList(
                [UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        # Actor 网络输出层：输出动作概率分布
        self.actor_head = nn.Sequential(
            nn.Softmax(dim=-1) # 输出概率分布
        )

        # Critic 网络输出层：输出状态价值
        self.critic_head = nn.Sequential(
            nn.Linear(2 * n_features, n_features), # 同样需要根据 ReadoutLayer 输出调整输入维度
            nn.ReLU(),
            nn.Linear(n_features, 1) # 输出单个状态价值
        )

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout) # 保留原有的 ReadoutLayer 作为共享部分

    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs, use_tensor_core=False):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        obs.transpose_(-1, -2)

        node_features = obs[:, :, 0:self.n_obs_in]
        adj = obs[:, :, self.n_obs_in:]
        if use_tensor_core:
            norm = self.get_normalisation(adj).half()
        else:
            norm = self.get_normalisation(adj)
        init_node_embeddings = self.node_init_embedding_layer(node_features)
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        current_node_embeddings = init_node_embeddings

        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj)
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                               edge_embeddings,
                                                                               norm,
                                                                               adj)

        shared_features = self.readout_layer(current_node_embeddings) # 共享的特征表示

        # 从共享特征中分别计算策略和价值
        action_probs = self.actor_head(shared_features)
        state_values = self.critic_head(shared_features)

        return action_probs, state_values
    
class EdgeAndNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features

        self.edge_embedding_NN = nn.Linear(int(n_obs_in + 1), n_features - 1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm):
        edge_features = torch.cat([adj.unsqueeze(-1),
                                   node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
                                  dim=-1)

        edge_features *= (adj.unsqueeze(-1) != 0).float()

        edge_features_unrolled = torch.reshape(edge_features, (edge_features.shape[0],
                                                               edge_features.shape[1] * edge_features.shape[1],
                                                               edge_features.shape[-1]))
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[1], self.n_features - 1))
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm

        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm.max()], dim=-1)))

        return edge_embeddings


class UpdateNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2 * n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2 * n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, norm, adj):
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings) / norm

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):

    def __init__(self, n_features, n_hid=[], bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid) != list:
            n_hid = [n_hid]

        n_hid = [2 * n_features] + n_hid + [1]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        f_local = node_embeddings

        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])
        f_pooled = h_pooled.repeat(1, 1, node_embeddings.shape[1]).view(node_embeddings.shape)

        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i < len(self.layers_readout) - 1:
                features = F.relu(features)
                #调试
                # out = features
            else:
                out = features

        return out

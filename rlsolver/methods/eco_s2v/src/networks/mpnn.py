import torch
import torch.nn as nn
import torch.nn.functional as F


class EECO_MPNN(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):

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

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs_in):
        """
        新输入 obs 的形状: (batch_size, n_graphs, n_envs, 27, 20)
        对应 (B, G, E, N, F)
        """
        obs = obs_in.clone()
        if obs.dim() == 4:
            obs = obs.unsqueeze(0)
        B, G, E, N, F = obs.shape  # batch_size, n_graphs, n_envs, 27, 20
        obs = obs.view(B * G * E, N, F)

        obs = obs.transpose(-1, -2)

        # node_features: 取前 self.n_obs_in (=7) 个channel作为节点特征
        node_features = obs[:, :, :self.n_obs_in]  # (B', F, N) -> 切片到 (B', 7, N)
        # adjacency: 取剩下的 channel 作为 adjacency
        adj = obs[:, :, self.n_obs_in:]  # (B', F, N) -> (B', F-7, N)

        # 计算邻接矩阵的归一化系数
        norm = self.get_normalisation(adj)

        # 初始节点嵌入 & 边的嵌入
        init_node_embeddings = self.node_init_embedding_layer(node_features)
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        # 多次迭代更新节点嵌入
        current_node_embeddings = init_node_embeddings
        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(
                    current_node_embeddings, edge_embeddings, norm, adj
                )
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](
                    current_node_embeddings, edge_embeddings, norm, adj
                )

        # 读出层
        out = self.readout_layer(current_node_embeddings)
        # 此时 out.shape 大概率是 (B' = B*G*E, N, 1) 或 (B' = B*G*E, N)

        # ========== Step 3: (可选) 还原到 (B, G, E, N, *) ==========
        out = out.view(B, G, E, -1)
        out = out.squeeze(0)
        return out


class MPNN(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 n_hid_readout=[], ):

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

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        obs.transpose_(-1, -2)

        # Calculate features to be used in the MPNN
        node_features = obs[:, :, 0:self.n_obs_in]

        # Get graph adj matrix.
        adj = obs[:, :, self.n_obs_in:]
        # adj_conns = (adj != 0).type(torch.FloatTensor).to(adj.device)

        norm = self.get_normalisation(adj)

        init_node_embeddings = self.node_init_embedding_layer(node_features)
        edge_embeddings = self.edge_embedding_layer(node_features, adj, norm)

        # Initialise embeddings.
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

        out = self.readout_layer(current_node_embeddings)
        out = out.squeeze()

        return out


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
            else:
                out = features

        return out

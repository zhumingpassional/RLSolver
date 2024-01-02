import os
import sys
import time
import tqdm
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from l2a_mh_sampling import BnMLP
from graph_utils import generate_graph, build_adjacency_matrix
from graph_utils import get_hot_tensor_of_graph

"""
self.trs_enc = nn.TransformerEncoder(...)
self.trs_enc(...)

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/8965#issuecomment-1530085758
pip3 install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
"""


class GraphTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(inp_dim, inp_dim, mid_dim, embed_dim), activation=nn.GELU())

        self.trs_encoders = []
        for layer_id in range(num_layers):
            trs_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoders.append(trs_encoder)
            setattr(self, f'trs_encoder{layer_id:02}', trs_encoder)
        self.mlp_enc = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.trs_decoders = []
        for layer_id in range(num_layers):
            trs_decoder = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_decoders.append(trs_decoder)
            setattr(self, f'trs_decoder{layer_id:02}', trs_decoder)
        self.mlp_dec = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.num_nodes = inp_dim
        self.mlp_node = BnMLP(dims=(embed_dim, mid_dim, mid_dim, embed_dim), activation=nn.GELU())
        self.mlp_matrix = BnMLP(dims=(embed_dim, mid_dim, mid_dim, mid_dim), activation=None)
        self.activation = nn.GELU()

        self.mlp0 = BnMLP(dims=(mid_dim + embed_dim, mid_dim, out_dim, out_dim), activation=None)

    def forward(self, inp0, mask):
        enc1, dec_matrix, dec_node = self.get_attn_matrix(inp0, mask)

        dec1 = [trs_decoder(enc1, dec_node, mask, mask) for trs_decoder in self.trs_decoders]
        dec1 = self.mlp_dec(th.concat(dec1, dim=2))

        dec = th.concat((dec_matrix.repeat(self.num_nodes, 1, 1), dec1), dim=2)
        out = self.mlp0(dec)
        return out

    def get_attn_matrix(self, inp0, mask):
        enc0 = self.mlp_inp(inp0)

        enc1 = [trs_encoder(enc0, mask) for trs_encoder in self.trs_encoders]
        enc1 = self.mlp_enc(th.concat(enc1, dim=2))

        dec_matrix = self.mlp_matrix(enc1).sum(dim=0, keepdims=True)
        dec_matrix = self.activation(dec_matrix)
        dec_node = self.mlp_node(enc1)
        return enc1, dec_matrix, dec_node


class PolicyTRS(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.mlp_inp = BnMLP(dims=(embed_dim + 1, mid_dim, embed_dim), activation=nn.GELU())

        self.trs_encoders = []
        for layer_id in range(num_layers):
            trs_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                     dropout=0., dim_feedforward=mid_dim, activation=nn.GELU())
            self.trs_encoders.append(trs_encoder)
            setattr(self, f'trs_encoder{layer_id:02}', trs_encoder)
        self.mlp_enc = BnMLP(dims=(embed_dim * num_layers, embed_dim * num_layers, embed_dim), activation=nn.GELU())

        self.mlp_matrix = BnMLP(dims=(embed_dim, mid_dim, mid_dim, mid_dim), activation=None)
        self.activation = nn.GELU()

        # self.mlp0 = BnMLP(dims=(embed_dim + mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())
        self.mlp0 = BnMLP(dims=(mid_dim + embed_dim + mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

        self.num_nodes = inp_dim

    def forward(self, prob, dec_node):
        # assert prob.shape == (num_nodes, )
        # assert dec_node.shape == (num_nodes, embed_dim)

        inp0 = th.concat((dec_node, prob[:, None]), dim=1)[:, None, :]
        # assert inp0.shape == (num_nodes, 1, embed_dim+1)
        enc0 = self.mlp_inp(inp0)
        # assert enc0.shape == (num_nodes, 1, embed_dim)

        enc1 = [trs_encoder(enc0, src_mask=None) for trs_encoder in self.trs_encoders]
        enc1 = self.mlp_enc(th.concat(enc1, dim=2))

        dec_prob = self.mlp_matrix(enc1).sum(dim=0)
        dec_prob = self.activation(dec_prob)
        # assert dec_prob.shape == (mid_dim, )
        return dec_prob

    def auto_regression(self, prob, dec_node, dec_matrix):
        i = 0
        dec_matrix = dec_matrix[:, i, :]
        dec_node = dec_node[:, i, :]
        # assert dec_node.shape == (num_nodes, batch_size, embed_dim)
        # assert dec_matrix.shape == (batch_size, mid_dim)

        dec_prob = self(prob.clone().detach(), dec_node.detach())
        dec = th.concat((dec_node, dec_matrix.repeat(self.num_nodes, 1), dec_prob.repeat(self.num_nodes, 1)), dim=1)
        prob = self.mlp0(dec).squeeze(1)
        return prob


def create_mask(seq_len, mask_type):
    if mask_type == 'triu':
        # Create an upper triangular matrix with ones above the diagonal
        mask = th.triu(th.ones(seq_len, seq_len), diagonal=1)
    elif mask_type == 'eye':
        # Create a square matrix with zeros on the diagonal
        mask = th.eye(seq_len)
    else:
        raise ValueError("type should in ['triu', 'eye']")
    return mask.bool()  # True means not allowed to attend.


def check_graph_trs_net():
    # Example usage:
    num_nodes = 100
    num_heads = 4
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads
    batch_size = 3

    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

    seq_len = num_nodes
    mask = create_mask(seq_len, mask_type='eye')
    input_tensor = th.rand(seq_len, batch_size, inp_dim)
    output_tensor = net(input_tensor, mask)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)


'''train graph_trs'''


def generate_adjacency_seq(num_sims, graph_type, num_nodes, if_tqdm=False):
    adjacency_seq = th.empty((num_nodes, num_sims, num_nodes), dtype=th.bool)

    i_iteration = tqdm.trange(num_sims, ascii=True) if if_tqdm else range(num_sims)
    for i in i_iteration:
        graph = generate_graph(graph_type=graph_type, num_nodes=num_nodes)
        adjacency_ary = build_adjacency_matrix(graph_list=graph, if_bidirectional=True)
        adjacency_seq[:, i, :] = adjacency_ary.eq(1)
    return adjacency_seq


def get_buffer(train_inp_path, buf_size, graph_type, num_nodes, device):
    train_inp = th.empty((num_nodes, buf_size, num_nodes), dtype=th.bool, device=device)
    if os.path.exists(train_inp_path):
        inp = th.load(train_inp_path, map_location=device)
        load_size = min(inp.shape[1], buf_size)
        train_inp[:, :load_size, :] = inp[:, :load_size, :]
    else:
        load_size = 0
    generate_size = buf_size - load_size
    if generate_size > 0:
        inp = generate_adjacency_seq(num_sims=generate_size, if_tqdm=True,
                                     graph_type=graph_type, num_nodes=num_nodes).to(device)
        train_inp[:, load_size:, :] = inp
    if buf_size > load_size:
        th.save(train_inp, train_inp_path)

    rand_is = th.randperm(buf_size, device=device)
    return train_inp[:, rand_is, :]


def train_graph_trs_net():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = ['erdos_renyi', 'powerlaw', 'barabasi_albert'][1]
    num_sims = 32

    buf_size = 2 ** 12
    num_epochs = buf_size // num_sims
    net_path = f"./attention_net_{graph_type}_Node{num_nodes}_{gpu_id}.pth"
    show_gap = 32

    if os.name == 'nt':
        num_sims = 8
        buf_size = num_sims * 4

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads
    learning_rate = 2 ** -12
    weight_decay = 0

    net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                   embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    th.save(net.state_dict(), net_path)

    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=False) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=False, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    '''buffer'''
    timer = time.time()
    mask = create_mask(seq_len=num_nodes, mask_type='eye').to(device)

    def get_objective(_adj_matrix, if_train):
        if if_train:
            net.train()
            _mask = mask
        else:
            net.eval()
            _mask = None

        inp = _adj_matrix.float()
        lab1 = th.empty_like(inp)
        lab2 = th.empty_like(inp)
        for i in range(lab1.shape[1]):
            adj = _adj_matrix[:, i, :]
            lab1[:, i, :] = get_hot_tensor_of_graph(adj_matrix=adj / int(num_nodes ** 0.5))
            lab2[:, i, :] = get_hot_tensor_of_graph(adj_matrix=adj / (adj.sum(dim=1, keepdim=True) * 0.25))

        out = net(inp.float(), mask=_mask)
        out1, out2 = th.chunk(out, chunks=2, dim=2)
        _objective = criterion(out1, lab1.detach()) + criterion(out2, lab2.detach())
        return _objective

    for buffer_id in list(range(8)) + list(range(8)) + list(range(9)):
        with th.no_grad():
            train_inp_path = f"./train_inp_{graph_type}_Node{num_nodes}_{buffer_id:02}.pth"
            train_inp = get_buffer(train_inp_path, buf_size, graph_type, num_nodes, device)

        '''train loop'''
        for j in range(num_epochs):
            j0 = j * num_sims
            j1 = j0 + num_sims
            adj_matrix = train_inp[:, j0:j1, :]

            '''valid'''
            if j % show_gap == 0:
                objective = get_objective(adj_matrix, if_train=False)

                exec_time = int(time.time() - timer)
                print(f"| {buffer_id:2}  {j:4}  {exec_time:4} sec | obj {objective.item():9.4f}")

            '''train'''
            objective = get_objective(adj_matrix, if_train=True)

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net_params, 3)
            optimizer.step()

    th.save(net.state_dict(), net_path)
    print(f"| save net in {net_path}")


def valid_graph_trs_net():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    graph_type = ['erdos_renyi', 'powerlaw', 'barabasi_albert'][1]
    num_sims = 16
    net_path = f"./attention_net_{graph_type}_Node{num_nodes}.pth"

    '''model'''
    num_heads = 8
    num_layers = 4
    inp_dim = num_nodes
    mid_dim = 256
    out_dim = num_nodes * 2
    embed_dim = int(inp_dim ** 0.5) - int(inp_dim ** 0.5) % num_heads

    graph_net = GraphTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim,
                         embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    graph_net.load_state_dict(th.load(net_path, map_location=device))

    inp = generate_adjacency_seq(num_sims=num_sims, graph_type=graph_type, num_nodes=num_nodes).to(device)
    dec_seq, dec_matrix, dec_node = graph_net.get_attn_matrix(inp.float(), mask=None)

    # from graph_utils import show_array2d
    # show_array2d(dec_node[:, 0, :].cpu().data.numpy())
    # show_array2d(dec_matrix[0, :, :].cpu().data.numpy())
    # print()

    policy_net = PolicyTRS(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=1,
                           embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to(device)
    prob = th.zeros(num_nodes, dtype=th.float32, device=device)
    prob = policy_net.auto_regression(prob, dec_node, dec_matrix)
    print()


if __name__ == '__main__':
    train_graph_trs_net()
    # valid_graph_trs_net()

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from typing import List
import math

from torch.nn.functional import normalize
from torch_geometric.nn import GCNConv, GATConv
from torchdyn.core import NeuralODE

# DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'
DEVICE = 'cpu'

class MLPTransform(nn.Module):
    def __init__(self, input_dim, hiddenunits: List[int], num_classes, prob_matrix, bias=True,
                 drop_prob=0.5, n_power_iterations=10, eps=1e-12, coeff=0.9):
        super(MLPTransform, self).__init__()
        self.features = None
        self.input_dim = input_dim
        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)

        fcs = [nn.Linear(input_dim, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i]))
        fcs.append(nn.Linear(hiddenunits[-1], num_classes))

        self.fcs = nn.ModuleList(fcs)

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = nn.Dropout(drop_prob)
        self.act_fn = nn.ReLU()
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.coeff = coeff

    def forward(self, nodes_vec):

        layer_inner = self.act_fn(self.fcs[0](self.dropout(nodes_vec)))
        for fc in self.fcs[1:-1]:
            weight = self.compute_weight(fc)
            fc.weight.data.copy_(weight)
            layer_inner = self.act_fn(fc(layer_inner))

        res = torch.relu(self.fcs[-1](self.dropout(layer_inner)))
        return res  # (nodes_num, 1)

    def compute_weight(self, module):
        device = next(module.parameters()).device
        weight = module.weight.clone()
        u = torch.rand(weight.shape[0], device=device)
        v = torch.rand(weight.shape[0], device=device)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps, out=v)
                u = normalize(torch.mv(weight, v), dim=0, eps=self.eps, out=u)
            if self.n_power_iterations > 0:
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        return weight


class GCN_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, adj, prob_matrix, dropout=0.0):
        super(GCN_Module, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, out_dim)
        self.input_dim = input_dim
        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)

        self.adj = adj
        # self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seed_vec):
        attr_mat = [seed_vec.T]
        for i in range(self.input_dim - 1):
            attr_mat.append(self.prob_matrix.T @ attr_mat[-1])
        attr_mat = torch.stack(attr_mat, dim=0)
        attr_mat = attr_mat.permute(2, 1, 0)
        x = F.relu(self.gc1(attr_mat, self.adj))
        x = self.dropout(x)
        x = F.relu(self.gc2(x, self.adj))
        #         x = self.dropout(x)
        #         x = F.relu(self.gc3(x, adj))
        '''
        # max pooling over nodes
        x = torch.max(x, dim=1)[0].squeeze()
        '''
        return x

class GAT_Module(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads, adj, prob_matrix, dropout=0.2):
        super(GAT_Module, self).__init__()
        self.prob_matrix = nn.Parameter((torch.FloatTensor(prob_matrix)), requires_grad=False)
        self.adj = adj
        self.input_dim = input_dim
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, out_dim, heads=1, dropout=dropout)

    def forward(self, seed_vec):
        seed_vec = seed_vec.transpose(0, 1)
        min_val, _ = torch.min(seed_vec, dim=0)
        max_val, _ = torch.max(seed_vec, dim=0)
        normalized_prob = (seed_vec - min_val) / (max_val - min_val)
        normalized_prob = normalized_prob.transpose(0, 1)
        seed_vec = normalized_prob.unsqueeze(-1)
        out = []
        for each_seed_vec in seed_vec:
            x = F.relu(self.conv1(each_seed_vec, self.adj))
            x = F.relu(self.conv2(x, self.adj))
            x = F.relu(self.conv3(x, self.adj))
            out.append(x)
        out = torch.stack(out, dim=0)

        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.bn = nn.BatchNorm1d(num_features=latent_dim)

    def forward(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        self.FC_input.weight.data.copy_(compute_weight(self.FC_input))
        h_ = self.FC_input(inputs)
        self.FC_input2.weight.data.copy_(compute_weight(self.FC_input2))
        h_ = self.FC_input2(h_)
        self.FC_mean.weight.data.copy_(compute_weight(self.FC_mean))
        mean = self.FC_mean(h_)
        self.FC_var.weight.data.copy_(compute_weight(self.FC_var))
        log_var = self.FC_var(h_)
        return mean, log_var


def compute_weight(module, n_power_iterations=10, eps=1e-12, coeff=0.9):
    device = next(module.parameters()).device
    weight = module.weight.clone()
    u = torch.rand(weight.shape[0], device=device)
    v = torch.rand(weight.shape[0], device=device)
    with torch.no_grad():
        for _ in range(n_power_iterations):
            v = normalize(torch.mv(weight.t(), u), dim=0, eps=eps, out=v)
            u = normalize(torch.mv(weight, v), dim=0, eps=eps, out=u)
        if n_power_iterations > 0:
            u = u.clone()
            v = v.clone()

    sigma = torch.dot(u, torch.mv(weight, v))
    factor = torch.max(torch.ones(1).to(weight.device), sigma / coeff)
    weight = weight / factor
    return weight


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h = F.relu(self.FC_input(inputs))
        h = F.relu(self.FC_hidden_1(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class CVAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(CVAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        # (batch_size, latent_dim)
        std = torch.exp(0.5 * var)
        epsilon = torch.randn_like(var)
        return mean + std * epsilon

    def forward(self, x, c, adj=None):
        if adj != None:
            mean, log_var = self.Encoder(x, adj)
        else:
            mean, log_var = self.Encoder(x, c)  # (batch_size, latent_dim)
        z = self.reparameterization(mean, log_var).to(x.device)
        x_hat = self.Decoder(z, c).to(x.device)

        return x_hat, mean, log_var  # (batch_size, latent_dim)

class NodeEncoder(nn.Module):
    """Node ODE function."""
    def __init__(self, input_dim, output_dim, dropout=0.9):
        super(NodeEncoder, self).__init__()

        self.input_dim = input_dim
        self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)
        self.gnn1 = GCNConv(input_dim, output_dim)

        self.w_node2edge = nn.Linear(input_dim * 2, input_dim)  # e_ij = W([x_i||x_j])
        self.init_network_weights(self.w_node2edge)
        self.w_edge2value = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.SiLU(),
            nn.Linear(self.input_dim // 2, 1))
        self.init_network_weights(self.w_edge2value)
    
    def init_network_weights(self, net, std = 0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)
    
    def set_x0(self, x_0):
        self.x_0 = x_0.clone().detach()

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index
        self.node_send, self.node_rec = edge_index[0], edge_index[1]

    def rel_rec_compute(self, node_inputs):
        # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        rel_send, rel_rec = node_inputs[self.node_send].to(DEVICE), node_inputs[self.node_rec].to(DEVICE)
        
        edges = torch.cat([rel_send, rel_rec], dim=-1)
        edges_from_node = F.silu(self.w_node2edge(edges))
        edges_z = self.dropout(edges_from_node)
        edge_2_value = torch.squeeze(F.sigmoid(self.w_edge2value(edges_z)), dim=-1)

        return edge_2_value
    

    def forward(self, node_inputs):
        inputs = self.layer_norm(node_inputs)
        edge_weight = self.rel_rec_compute(inputs)
        x_hidden = self.gnn1(inputs, self.edge_index, edge_weight)
        x_hidden = self.dropout(x_hidden)

        # return x_hidden - inputs
        return x_hidden

class CoupledODEFunc(nn.Module):
    def __init__(self, node_ode_func_net, node_size, dropout_rate):
        super(CoupledODEFunc, self).__init__()

        self.node_size = node_size
        self.dropout = nn.Dropout(dropout_rate)

        self.node_ode_func_net = node_ode_func_net  # input: node_embedding, edge_index, edge_weight

        self.alpha = nn.Parameter(0.8 * torch.ones(self.node_size))
        self.w = nn.Parameter(torch.eye(self.node_ode_func_net.input_dim))
        self.d = nn.Parameter(torch.ones(self.node_ode_func_net.input_dim))

    def set_edge_index(self, edge_index):
        self.node_ode_func_net.set_edge_index(edge_index)
    
    def init_network_weights (self, net, std=0.1):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.constant_(m.bias, val=0)

    def forward(self, input_embedding):
        # hyper and checkers
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1)
        assert (not torch.isnan(input_embedding).any())

        node_attributes = input_embedding
        grad_node = self.node_ode_func_net(node_attributes)  # [K*N,D]
        assert (not torch.isnan(grad_node).any())

        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        grad_node2 = torch.einsum('il, lm->im', input_embedding, w)
        grad_node2 = self.dropout(grad_node2)

        return alpha / 2 * grad_node - input_embedding + grad_node2 - input_embedding


class DiffusionODE(nn.Module):
    def __init__(self, edge_index, node_size, input_dim = 1, out_dim = 5, t_size = 2, dropout_rate = 0.1):
        super(DiffusionODE, self).__init__()

        self.node_size = node_size
        self.edge_index = edge_index
        self.t_size = t_size
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.d_model = out_dim
        self.t_span = torch.linspace(0, 1, t_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.input_Lin = nn.Linear(self.input_dim, self.input_dim*2)
        self.output_Lin = nn.Linear(self.input_dim*2, self.input_dim)
        self.output_Lin = nn.Linear(self.input_dim*2, self.out_dim)

        self.dy_att = nn.Parameter(torch.ones(1, self.d_model))
        self.dy_att_m = nn.Parameter(torch.zeros(self.d_model, self.d_model))
        init.xavier_normal_(self.dy_att.data)
        init.xavier_normal_(self.dy_att_m.data)

        ## 1. Node ODE function
        self.node_ode_func_net = NodeEncoder(input_dim=self.input_dim * 2, output_dim=self.input_dim * 2, dropout=0.8)

        self.NeuralFunc =  CoupledODEFunc(
            node_ode_func_net = self.node_ode_func_net,
            node_size = self.node_size, dropout_rate=0.1)
        self.neuralDE = NeuralODE(self.NeuralFunc, solver='rk4') #

    def DyEmbeddingMerger(self, all_embeddings):
        all_embeddings = self.channel_attention(*all_embeddings)
        return all_embeddings

    def channel_attention(self, *channel_embeddings):

        weights = []
        for embedding in channel_embeddings:
            weights.append(
                torch.sum(torch.multiply(self.dy_att, torch.matmul(embedding, self.dy_att_m)),1))
        embs = torch.stack(weights, dim=0)
        score = F.softmax(embs.t(), dim=-1)
        mixed_embeddings = 0
        for i in range(len(weights)):
            mixed_embeddings += torch.multiply(score.t()[i], channel_embeddings[i].t()).t()
        return mixed_embeddings

    def forward(self, preds):
        all_graph_embeddings = []
        self.NeuralFunc.set_edge_index(self.edge_index)
        for i in range(len(preds)):
            graph_x_embeddings = self.input_Lin(preds[i])
            graph_x_embeddings = F.normalize(graph_x_embeddings, dim=-1, p=2)
            graph_x_embeddings = self.dropout(graph_x_embeddings)

            _, graph_x_embeddings = self.neuralDE(graph_x_embeddings, t_span=self.t_span)

            output_graph_embeddings = []
            for i in range(1, self.t_size):
                user_dynamic_embedding = self.output_Lin(F.silu(graph_x_embeddings[i])).to(DEVICE)
                output_graph_embeddings.append(user_dynamic_embedding)
            merge_embedding = self.DyEmbeddingMerger(output_graph_embeddings)
            all_graph_embeddings.append(merge_embedding)
        return torch.stack(all_graph_embeddings, dim=0).squeeze(dim=-1)




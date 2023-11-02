import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch.nn.init import xavier_normal_
from torch.nn import Parameter

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


class RGCN(nn.Module):
    def __init__(self, n_entities, n_relations, params):
        super(RGCN, self).__init__()
        
        self.params = params
        self.num_ent = n_entities
        self.n_relations = n_relations
        torch.manual_seed(12345)
        self.device = params.device

        self.node_emb = get_param((self.num_ent, self.params.gcn_dim_init)).to(self.params.device)
        
        self.conv_layers = torch.nn.ModuleList()
        
        self.conv_layers.append(
            RGCNConv(self.params.gcn_dim_init, self.params.hidden_channels_gcn, num_relations=n_relations))
        
        for _ in range(self.params.conv_layers - 2):
            self.conv_layers.append(
            RGCNConv(self.params.hidden_channels_gcn, self.params.hidden_channels_gcn, num_relations=n_relations))
        
        self.input_lin = nn.Linear(self.params.hidden_channels_gcn, self.params.hidden_channels_gcn, bias=True)

    
    def forward(self, node_index, edge_index, edge_type):
        x = self.node_emb[node_index].to(self.device)
        
        if self.params.transform_input:
            x = self.input_lin(x)
        
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, self.params.drop_gcn, training=self.training)

        
        return x

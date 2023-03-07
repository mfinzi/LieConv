import torch_geometric
import torch

import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


class LieGNNSimpleConv(MessagePassing):
    """
        Perform simple equivariant convolution:

        h_u = \phi (\sum_v d((u_i, q_i), (v_j, q_j)) * h^{l-1}_v)

        where d^2 = ||log(v^{-1}u)||^2 + \alpha ||q_i - q_j||^2
        (the default distance in the LieConv paper)
    """
    def __init__(self, c_in, c_out, hidden_dim=None, agg='add', **kwargs):
        super().__init__(aggr=agg)
        if hidden_dim is None:
            hidden_dim = c_out
        self.mlp = nn.Sequential(
            nn.Linear(c_in, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, c_out), 
            nn.BatchNorm1d(c_out), nn.ReLU()
        ) 
    
    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, edge_attr):
        """
        x_i: (e, d_h) values at the source node i
        edge_attr: (e, d_e) edge attributes
        Calculate product of distance and hidden representations
        """
        messages = torch.einsum("ij,ik->ij", x_i, edge_attr)
        return messages
    
    def aggregate(self, inputs, index):
        """
        Aggregate messages from all neighbouring nodes
        
        inputs: (e, d_h) messages m_ij for each node
        index: (e, 1) source nodes for each message
        """
        aggr_out = scatter(inputs,
                           index,
                           dim=self.node_dim, 
                           reduce=self.aggr) 
        return aggr_out

    def update(self, aggr_out):
        """
        Apply MLP to the convolved values
        aggr_out: (n, d_h) convolved values
        """
        return self.mlp(aggr_out)

class LieConvMPGNN(MessagePassing):
    def __int__(self):
        pass

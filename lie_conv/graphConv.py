import torch_geometric
import torch

import torch.nn as nn
from torch_geometric.nn import MessagePassing

class LieConvSimpleConv(SimpleConv):
    """
        Perform simple equivariant convolution:

        h_u = \phi (\sum_v d((u_i, q_i), (v_j, q_j)) * h^{l-1}_v)

        where d^2 = ||log(v^{-1}u)||^2 + \alpha ||q_i - q_j||^2
        (the default distance in the LieConv paper)
    """
    def __init__(self, c_in, c_out, agg='add'):
        super.__init__(aggr=agg)

        self.mlp = nn.Sequential([
            nn.Linear(c_in, c_out), BatchNorm1d(c_out), ReLU(),
            nn.Linear(c_out, c_int), BatchNorm1d(c_out), ReLU()
        ]) 
    
    def forward(self, x, edge_index, edge_weight):
        # Convolve values using the pre-generated distances
        y = super.forward(x=x, 
                          edge_index=edge_index, 
                          edge_weight=edge_weight)

        # Apply MLP to the convolved values
        return self.mlp(y) 

class LieConvMPGNN(MessagePassing):
    def __int__(self):
        pass

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GATConv

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, out_dim, num_heads=1)

    def forward(self, g, features):
        # Layer 1
        total_node_num = features.shape[0]
        h = self.conv1(g, features)
        h = F.elu(h)
        # Layer 2
        h = h.reshape(total_node_num, self.hidden_dim * self.num_heads)
        h = self.conv2(g, h).mean(1)  # Aggregate the outputs across all nodes
        return h
'''
class Model(nn.Module):
    def __init__(self, test=True):
        self.test = test

    def getGraph(sequence):
        if self.test:
            return predict_graph(sequence)
        else:
            return torch.load(f'RFold/predicted_graph/{sequence}.pth')

    def forward(self, sequences): # Note change by Kaya - Now accepts sequences for batch parallelism
        graph = getGraph(sequence)  # TODO handle batch

        #TODO Gyu


        # Return Format: List [ Tensor [ 2 (types of reagents) x Length of Sequence] for each sequence in sequences]
'''
